import os
import re
import pandas as pd
import requests
from google.cloud import storage
import random
import logging
from typing import Dict, Optional, List
import time
from decimal import Decimal, ROUND_HALF_UP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ----------------------------
# Helpers (shared)
# ----------------------------
IMG_RE = re.compile(r'(https?://[^\s",>]+?\.(?:jpg|jpeg|png|webp))', re.IGNORECASE)


def extract_image_series(df: pd.DataFrame, col_name: str) -> pd.Series:
    """
    Always returns a pandas Series of strings (no ndarray).
    Performs row-wise regex extraction of a direct image URL.
    """
    if col_name not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    return df[col_name].astype(str).apply(
        lambda s: (IMG_RE.search(s).group(0) if IMG_RE.search(s) else "")
    )


def parse_money_to_float(series: pd.Series) -> pd.Series:
    """
    Parse money-like strings to floats. Strips currency words/symbols, commas,
    stray chars; collapses multiple decimals.
    """
    def _clean(v):
        if pd.isna(v):
            return None
        s = str(v).strip()
        s = s.replace(",", "")
        s = re.sub(
            r"(usd|cad|inr|aud|eur|gbp|jpy|chf|sek|mxn|brl|sgd|hkd|nzd|dkk|nok|£|€|\$)",
            "",
            s,
            flags=re.IGNORECASE,
        )
        s = re.sub(r"[^0-9.\-]", "", s)
        if s.count(".") > 1:
            parts = s.split(".")
            s = parts[0] + "." + "".join(parts[1:])
        try:
            return float(s) if s != "" else None
        except Exception:
            return None

    out = series.map(_clean)
    return pd.to_numeric(out, errors="coerce")


class DiamondCatalogProcessor:
    """Pinterest feed generator with robust parsing, pricing, and country files."""

    def __init__(self):
        # Directories
        self.download_dir = os.environ.get("DOWNLOAD_DIR", "/tmp/raw")
        self.output_folder = os.environ.get("OUTPUT_FOLDER", "/tmp/pinterest_output")

        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)

        # Nivoda direct download URLs (overridable by env for safety)
        self.download_urls = {
            "natural": os.environ.get(
                "NIVODA_NATURAL_URL",
                "https://g.nivoda.com/feeds-api/ftpdownload/1268df58-e992-455c-8be9-bb9b8abeeea0",
            ),
            "lab_grown": os.environ.get(
                "NIVODA_LABGROWN_URL",
                "https://g.nivoda.com/feeds-api/ftpdownload/4e87471c-ae7c-443d-a923-ee51a510b5fa",
            ),
            "gemstone": os.environ.get(
                "NIVODA_GEMSTONE_URL",
                "https://g.nivoda.com/feeds-api/ftpdownload/1dddf473-d048-48ec-97bb-971cfae092b6",
            ),
        }

        # Local file paths for the downloaded CSVs
        self.source_files = {
            "natural": {
                "url": self.download_urls["natural"],
                "local_path": os.path.join(self.download_dir, "Natural.csv"),
            },
            "lab_grown": {
                "url": self.download_urls["lab_grown"],
                "local_path": os.path.join(self.download_dir, "Labgrown.csv"),
            },
            "gemstone": {
                "url": self.download_urls["gemstone"],
                "local_path": os.path.join(self.download_dir, "gemstones.csv"),
            },
        }

        # Google Cloud Storage configuration
        self.gcs_config = {
            "bucket_name": os.environ.get("BUCKET_NAME", "sitemaps.leeladiamond.com"),
            "bucket_folder": os.environ.get("BUCKET_FOLDER", "pinterestfinal"),
        }

        # Exchange rate API configuration
        self.exchange_rate_api_key = os.environ.get("EXCHANGE_RATE_API_KEY", "20155ba28afe7c763416cc23")

        # Country to currency mapping
        self.country_currency = {
            "US": "USD", "CA": "CAD", "GB": "GBP", "AU": "AUD",
            "DE": "EUR", "FR": "EUR", "IT": "EUR", "ES": "EUR",
            "NL": "EUR", "SE": "SEK", "CH": "CHF", "JP": "JPY",
            "MX": "MXN", "BR": "BRL", "IN": "INR", "SG": "SGD",
            "HK": "HKD", "NZ": "NZD", "DK": "DKK", "NO": "NOK"
        }

        # Pinterest-specific Google product categories for jewelry
        self.jewelry_categories: List[int] = [
            189, 190, 191, 197, 192, 194, 6463, 196, 200,
            5122, 5123, 7471, 6870, 201, 502979, 6540,
            6102, 5996, 198, 5982
        ]

    # ----------------------------
    # Download via HTTPS (no FTP)
    # ----------------------------
    def download_file_from_web(self, label: str, url: str, local_path: str, max_retries: int = 3) -> bool:
        for attempt in range(max_retries):
            try:
                logger.info(f"[HTTP] Downloading {label} from {url}")
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                with open(local_path, "wb") as f:
                    f.write(resp.content)
                logger.info(f"[HTTP] Saved {label} → {local_path}")
                return True
            except Exception as e:
                logger.warning(f"[HTTP] Attempt {attempt + 1} failed for {label}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"[HTTP] Failed to download {label} after {max_retries} attempts")
        return False

    def download_all_files(self) -> bool:
        logger.info("Starting HTTPS download process from Nivoda direct links...")
        success_count = 0
        for product_type, file_info in self.source_files.items():
            if self.download_file_from_web(product_type, file_info["url"], file_info["local_path"]):
                success_count += 1
            else:
                logger.error(f"Failed to download {product_type} file")
        logger.info(f"Downloaded {success_count}/{len(self.source_files)} files successfully")
        return success_count == len(self.source_files)

    # ----------------------------
    # Pricing / Currency
    # ----------------------------
    @staticmethod
    def usd_to_cad_rate() -> float:
        try:
            return float(os.environ.get("USD_TO_CAD", "1.41"))
        except Exception:
            return 1.41

    def convert_usd_to_cad_value(self, price_usd) -> float:
        """Convert a single USD value to CAD with configured rate."""
        rate = self.usd_to_cad_rate()
        try:
            return round(float(price_usd) * rate, 2)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def format_price(amount: float, currency: str) -> str:
        """
        Return price formatted as '123.45 CAD' (ISO-4217 with a space).
        Ensures two decimals and no scientific notation.
        """
        try:
            amt = float(amount)
        except Exception:
            amt = 0.0
        return f"{amt:.2f} {currency}"

    # ----------------------------
    # Cleaning / Validation
    # ----------------------------
    def clean_and_validate_data(self, df: pd.DataFrame, product_type: str) -> pd.DataFrame:
        logger.info(f"Cleaning {product_type} data - {len(df)} rows")
        df = df.fillna('')

        # Require valid image for Pinterest
        img_series = extract_image_series(df, 'image') if 'image' in df.columns else pd.Series([''] * len(df))
        before = len(df)
        df = df[img_series.str.len() > 0].copy()
        df['image'] = extract_image_series(df, 'image')
        logger.info(f"Filtered {before - len(df)} rows without valid images")

        # Clean text fields
        for col in ['shape', 'col', 'clar', 'cut', 'pol', 'symm', 'lab']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()

        # Validate required fields based on product type
        required_fields = {
            'natural': ['ReportNo', 'shape', 'carats', 'col', 'clar', 'lab'],
            'lab_grown': ['ReportNo', 'shape', 'carats', 'col', 'clar', 'lab'],
            'gemstone': ['ReportNo', 'shape', 'carats', 'Color', 'gemType'],
        }
        if product_type in required_fields:
            for field in required_fields[product_type]:
                if field in df.columns:
                    df = df[df[field].astype(str).str.len() > 0]

        logger.info(f"After cleaning: {len(df)} rows remaining")
        return df

    # ----------------------------
    # Pricing logic per product_type (smart fallbacks)
    # ----------------------------
    def compute_price_cad(self, df: pd.DataFrame, product_type: str) -> pd.Series:
        """
        Detect price with ordered fallbacks:
          1) markup_price / markupPrice
          2) delivered_price / deliveredPrice
          3) price
          4) price_per_carat * carats (or pricePerCarat * carats)
        Currency from markup_currency / markupCurrency (default USD).
        Returns CAD series.
        """
        rate = self.usd_to_cad_rate()

        if product_type == "lab_grown":
            price_candidates = ["markupPrice", "deliveredPrice", "price"]
            ppc_col = "pricePerCarat"
            carats_col = "carats"
            currency_col = "markupCurrency"
        elif product_type == "natural":
            price_candidates = ["markup_price", "delivered_price", "price"]
            ppc_col = "price_per_carat"
            carats_col = "carats"
            currency_col = "markup_currency"
        else:  # gemstone
            price_candidates = ["markup_price", "price", "price_per_carat"]
            ppc_col = "price_per_carat"
            carats_col = "carats"
            currency_col = "markup_currency"

        # Ensure referenced columns exist
        for c in set(price_candidates + [ppc_col, carats_col, currency_col]):
            if c and c not in df.columns:
                df[c] = ""

        used_source = pd.Series([""] * len(df), index=df.index)
        price_usd = pd.Series([None] * len(df), index=df.index, dtype="float64")

        # Try direct price candidates
        for cand in price_candidates:
            series = parse_money_to_float(df[cand]) if cand in df.columns else pd.Series([None] * len(df))
            take = price_usd.isna() & series.notna() & (series > 0)
            price_usd.loc[take] = series.loc[take]
            used_source.loc[take] = cand

        # Fallback: price_per_carat * carats
        missing = price_usd.isna() | (price_usd <= 0)
        if missing.any():
            ppc = parse_money_to_float(df[ppc_col]) if ppc_col in df.columns else pd.Series([None] * len(df))
            carats = pd.to_numeric(df[carats_col], errors="coerce")
            ppc_total = (ppc * carats).where(ppc.notna() & carats.notna(), other=None)
            take_ppc = missing & ppc_total.notna() & (ppc_total > 0)
            price_usd.loc[take_ppc] = ppc_total.loc[take_ppc]
            used_source.loc[take_ppc] = f"{ppc_col}*{carats_col}"

        # Currency handling
        if currency_col in df.columns:
            curr = df[currency_col].astype(str).str.strip().str.upper().replace({"": "USD"})
        else:
            curr = pd.Series(["USD"] * len(df), index=df.index)

        # Convert to CAD
        def to_cad(p, c):
            if p is None or pd.isna(p) or p <= 0:
                return 0.0
            if c == "CAD":
                return round(float(p), 2)
            # everything else treated as USD
            return round(float(p) * rate, 2)

        price_cad = pd.Series([to_cad(p, c) for p, c in zip(price_usd, curr)], index=df.index)
        price_cad = pd.to_numeric(price_cad, errors="coerce").fillna(0.0)

        logger.info(
            f"[PRICE][{product_type}] rows={len(df)} | "
            f"priced(>0 CAD)={(price_cad > 0).sum()} | "
            f"sources={used_source.value_counts(dropna=False).to_dict()}"
        )
        return price_cad

    # ----------------------------
    # Processing
    # ----------------------------
    def process_file(self, file_path: str, product_type: str) -> pd.DataFrame:
        try:
            logger.info(f"Processing {product_type} file: {file_path}")

            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                logger.error(f"Missing or empty file: {file_path}")
                return pd.DataFrame()

            # Robust CSV read
            try:
                df = pd.read_csv(file_path, dtype=str, low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, dtype=str, low_memory=False, encoding='utf-8-sig')

            if df.empty:
                logger.warning(f"No data in {product_type} file")
                return pd.DataFrame()

            # Ensure common columns exist
            base_cols = [
                "shape","carats","col","clar","cut","pol","symm","flo","floCol",
                "length","width","height","depth","table","culet","lab","girdle",
                "ReportNo","image","video","pdf","diamondId","stockId","ID"
            ]
            gemstone_only = [
                "gemType","Treatment","Mine of Origin","mine_of_origin","mineOfOrigin",
                "pdfUrl","price_per_carat","markup_price","markup_currency"
            ]
            natural_only  = [
                "price_per_carat","markup_price","markup_currency","mine_of_origin",
                "canada_mark_eligible","is_returnable","delivered_price"
            ]
            lab_only      = [
                "pricePerCarat","markupPrice","markupCurrency","deliveredPrice",
                "minDeliveryDays","maxDeliveryDays","mineOfOrigin","price"
            ]

            ensure_cols = set(base_cols + gemstone_only + natural_only + lab_only)
            for c in ensure_cols:
                if c not in df.columns:
                    df[c] = ""

            # Clean & validate (also enforces image existence)
            df = self.clean_and_validate_data(df, product_type)
            if df.empty:
                logger.warning(f"No valid data remaining in {product_type} file after cleaning")
                return pd.DataFrame()

            # Price pipeline (CAD)
            df["price_cad"] = self.compute_price_cad(df, product_type)

            # Drop rows with zero/invalid price to satisfy Pinterest constraint
            before_pr = len(df)
            df = df[df["price_cad"] > 0].copy()
            dropped = before_pr - len(df)
            if dropped:
                logger.info(f"Dropped {dropped} rows without valid price ({product_type})")

            # Format for Pinterest: '123.45 CAD'
            df["price"] = df["price_cad"].map(lambda x: self.format_price(x, "CAD"))

            # Generate unique IDs (fallbacks for gemstones)
            df["ReportNo"] = df["ReportNo"].astype(str).str.strip()
            if product_type == "gemstone" and (df["ReportNo"] == "").all():
                df["ReportNo"] = df["ID"].astype(str)
            df["id"] = (
                df["ReportNo"].where(df["ReportNo"] != "", df["diamondId"].astype(str)) + "CA"
            ).fillna("")

            # Product info (titles/desc/links)
            df = self.apply_product_templates(df, product_type)

            # Pinterest fields
            df = self.add_pinterest_fields(df)

            # Final safety: ensure valid price string
            df = df[
                df["price"]
                .astype(str)
                .str.match(r'^\d+\.\d{2}\s[A-Z]{3}$', na=False)
            ]

            logger.info(f"Successfully processed {len(df)} {product_type} products")
            return df

        except Exception as e:
            logger.error(f"Error processing {product_type} file {file_path}: {e}")
            return pd.DataFrame()

    def apply_product_templates(self, df: pd.DataFrame, product_type: str) -> pd.DataFrame:
        def meas(row):
            L = row.get("length")
            W = row.get("width")
            H = row.get("height") or row.get("depth")
            if L and W and H:
                return f"{L}-{W}x{H} mm"
            if L and W:
                return f"{L}x{W} mm"
            return ""

        product_templates = {
            "natural": {
                "title": lambda row: (
                    f"{row.get('shape', 'DIAMOND')} {row.get('carats', '')} Carats "
                    f"{row.get('col', '')} Color {row.get('clar', '')} Clarity "
                    f"{row.get('lab', '')} Certified Natural Diamond"
                ),
                "description": lambda row: (
                    f"Natural {row.get('shape','')} diamond: {row.get('carats','')} carats, "
                    f"{row.get('col','')} color, {row.get('clar','')} clarity. "
                    f"Measurements: {meas(row)}. Cut: {row.get('cut','')}, Polish: {row.get('pol','')}, Symmetry: {row.get('symm','')}, "
                    f"Table: {row.get('table','')}%, Depth: {row.get('depth','')}%, Fluorescence: {row.get('flo','') or row.get('floCol','')}. "
                    f"{row.get('lab','')} certified."
                ),
                "link": lambda row: (
                    f"https://leeladiamond.com/pages/natural-diamond-catalog?id={row.get('ReportNo', '')}"
                ),
            },
            "lab_grown": {
                "title": lambda row: (
                    f"{row.get('shape','')} {row.get('carats','')} Carats "
                    f"{row.get('col','')} Color {row.get('clar','')} Clarity "
                    f"{row.get('lab','')} Certified Lab Grown Diamond"
                ),
                "description": lambda row: (
                    f"Lab-grown {row.get('shape','')} diamond: {row.get('carats','')} carats, "
                    f"{row.get('col','')} color, {row.get('clar','')} clarity. "
                    f"Measurements: {meas(row)}. Cut: {row.get('cut','')}, Polish: {row.get('pol','')}, Symmetry: {row.get('symm','')}, "
                    f"Table: {row.get('table','')}%, Depth: {row.get('depth','')}%, Fluorescence: {row.get('flo','') or row.get('floCol','')}. "
                    f"{row.get('lab','')} certified."
                ),
                "link": lambda row: (
                    "https://leeladiamond.com/pages/lab-grown-diamonds/"
                    f"{str(row.get('shape','')).strip().lower()}-"
                    f"{str(row.get('carats','')).replace('.', '-')}-carat-"
                    f"{str(row.get('col','')).strip().lower()}-color-"
                    f"{str(row.get('clar','')).strip().lower()}-clarity-"
                    f"{str(row.get('lab','')).strip().lower()}-certified-"
                    f"{str(row.get('ReportNo','')).strip()}"
                ),
            },
            "gemstone": {
                "title": lambda row: (
                    f"{row.get('gemType', 'Gemstone')} {row.get('Color', '')} "
                    f"{row.get('shape', '')} – {row.get('carats', '')} Carats, "
                    f"{row.get('Clarity', '')} Clarity, {row.get('Cut', '')} Cut, "
                    f"{row.get('Lab', '')} Certified"
                ),
                "description": lambda row: (
                    f"{row.get('gemType','')} gemstone in {row.get('Color','')} color, "
                    f"{row.get('carats','')} carats. Measurements: {meas(row)}. "
                    f"Clarity: {row.get('Clarity','')}, Cut: {row.get('Cut','')}, Lab: {row.get('Lab','')}. "
                    f"Treatment: {(row.get('Treatment','') or 'N/A')}. "
                    f"{'Origin: ' + row.get('Mine of Origin','') if row.get('Mine of Origin','') else ''}"
                ),
                "link": lambda row: (
                    "https://leeladiamond.com/pages/gemstone-catalog?"
                    f"id={row.get('ReportNo', '') or row.get('ID','')}"
                ),
            },
        }

        if product_type in product_templates:
            template = product_templates[product_type]
            df['title'] = df.apply(template['title'], axis=1)
            df['description'] = df.apply(template['description'], axis=1)
            df['link'] = df.apply(template['link'], axis=1)

        df['title'] = df['title'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
        df['description'] = df['description'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
        return df

    # ----------------------------
    # Pinterest fields
    # ----------------------------
    def add_pinterest_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        num_rows = len(df)
        df['image_link'] = df['image']
        df['availability'] = 'in stock'
        df['google_product_category'] = [
            random.choice(self.jewelry_categories) for _ in range(num_rows)
        ]
        df['average_review_rating'] = [
            round(random.uniform(4.0, 5.0), 1) for _ in range(num_rows)
        ]
        df['number_of_ratings'] = [random.randint(5, 50) for _ in range(num_rows)]
        df['condition'] = 'new'
        return df

    # ----------------------------
    # Exchange rates & Country files
    # ----------------------------
    def get_exchange_rates(self) -> Optional[Dict]:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                url = f"https://v6.exchangerate-api.com/v6/{self.exchange_rate_api_key}/latest/USD"
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("result") == "success":
                        logger.info("Successfully fetched exchange rates")
                        return data["conversion_rates"]
                    else:
                        logger.error(
                            f"Exchange rate API error: "
                            f"{data.get('error-type', 'Unknown error')}"
                        )
                else:
                    logger.error(
                        f"HTTP error fetching exchange rates: {response.status_code}"
                    )
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout fetching exchange rates (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"Error fetching exchange rates (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        logger.error("Failed to fetch exchange rates, using fallback rates")
        return {
            "USD": 1.0, "CAD": 1.35, "GBP": 0.79, "EUR": 0.85, "AUD": 1.50,
            "JPY": 110.0, "CHF": 0.92, "SEK": 8.5, "MXN": 20.0, "BRL": 5.2,
            "INR": 74.0, "SGD": 1.35, "HKD": 7.8, "NZD": 1.60, "DKK": 6.3, "NOK": 8.8
        }

    def _parse_price_number(self, value: str) -> float:
        if value is None:
            return 0.0
        s = str(value).strip()
        m = re.search(r'(\d+(?:[.,]\d+)*)', s)
        if not m:
            return 0.0
        num = m.group(1).replace(',', '')
        try:
            return float(num)
        except ValueError:
            return 0.0

    def create_country_specific_files(self, combined_df: pd.DataFrame) -> None:
        logger.info("Creating country-specific files...")
        rates = self.get_exchange_rates()
        if not rates:
            logger.error("No exchange rates available, skipping country-specific file creation")
            return

        # USD->CAD from API (for correct CAD->target conversion)
        usd_to_cad = Decimal(str(rates.get("CAD", 1))) or Decimal("1")
        files_created = 0

        for country, currency in self.country_currency.items():
            try:
                if currency not in rates:
                    logger.warning(
                        f"Exchange rate not available for {currency}, skipping {country}"
                    )
                    continue

                country_data = combined_df.copy()
                country_data['id'] = country_data['id'].str.replace(
                    r'CA$', country, regex=True
                )

                # Convert CAD amounts to target currency: (target_per_USD / CAD_per_USD)
                target_rate = Decimal(str(rates[currency]))
                cad_to_target = (target_rate / usd_to_cad)

                country_data['price_numeric'] = country_data['price'].apply(
                    self._parse_price_number
                )
                country_data['price_numeric'] = country_data['price_numeric'].apply(
                    lambda x: float(
                        (Decimal(str(x)) * cad_to_target).quantize(
                            Decimal("0.01"), rounding=ROUND_HALF_UP
                        )
                    )
                )
                country_data['price'] = country_data['price_numeric'].map(
                    lambda x: self.format_price(x, currency)
                )
                country_data.drop(columns=['price_numeric'], inplace=True, errors='ignore')

                # Final guard: keep only rows with valid price format
                country_data = country_data[
                    country_data['price']
                    .astype(str)
                    .str.match(r'^\d+\.\d{2}\s[A-Z]{3}$', na=False)
                ]

                output_file = os.path.join(
                    self.output_folder, f"{country}-pinterest-csv.csv"
                )
                country_data.to_csv(output_file, index=False, encoding='utf-8')

                logger.info(f"Created {country} file with {len(country_data)} products")
                files_created += 1
            except Exception as e:
                logger.error(f"Error creating file for {country}: {e}")

        logger.info(f"Created {files_created} country-specific files")

    # ----------------------------
    # GCS Upload
    # ----------------------------
    def upload_to_gcs(self) -> bool:
        try:
            logger.info("Starting GCS upload...")
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.gcs_config["bucket_name"])

            files_uploaded = 0
            for file_name in os.listdir(self.output_folder):
                file_path = os.path.join(self.output_folder, file_name)
                if os.path.isfile(file_path):
                    try:
                        destination_blob_name = (
                            f"{self.gcs_config['bucket_folder'].rstrip('/')}/{file_name}"
                        )
                        blob = bucket.blob(destination_blob_name)
                        blob.upload_from_filename(file_path)
                        logger.info(f"Uploaded {file_name} to GCS")
                        files_uploaded += 1
                    except Exception as e:
                        logger.error(f"Error uploading {file_name}: {e}")

            logger.info(f"Successfully uploaded {files_uploaded} files to GCS")
            return files_uploaded > 0
        except Exception as e:
            logger.error(f"GCS upload error: {e}")
            return False

    # ----------------------------
    # Orchestration
    # ----------------------------
    def run(self) -> bool:
        try:
            logger.info("Starting Diamond Catalog Processing...")

            if not self.download_all_files():
                logger.error("Failed to download required files from Nivoda direct links")
                return False

            dataframes = []
            for product_type, file_info in self.source_files.items():
                df = self.process_file(file_info["local_path"], product_type)
                if not df.empty:
                    dataframes.append(df)
                else:
                    logger.warning(f"No data processed for {product_type}")

            if not dataframes:
                logger.error("No data was processed from any files")
                return False

            combined_df = pd.concat(dataframes, ignore_index=True)

            final_columns = [
                'id', 'title', 'description', 'link', 'image_link', 'price',
                'availability', 'google_product_category', 'average_review_rating',
                'number_of_ratings', 'condition'
            ]
            for col in final_columns:
                if col not in combined_df.columns:
                    combined_df[col] = ''
            combined_df = combined_df[final_columns]

            # Final enforcement: keep only rows with a valid ISO-4217 price string
            before_final = len(combined_df)
            combined_df = combined_df[
                combined_df['price']
                .astype(str)
                .str.match(r'^\d+\.\d{2}\s[A-Z]{3}$', na=False)
            ]
            after_final = len(combined_df)
            if before_final != after_final:
                logger.info(
                    f"Removed {before_final - after_final} rows lacking a valid formatted price"
                )

            combined_file = os.path.join(self.output_folder, "combined_catalog.csv")
            combined_df.to_csv(combined_file, index=False, encoding='utf-8')
            logger.info(f"Combined catalog saved with {len(combined_df)} total products")

            self.create_country_specific_files(combined_df)

            if self.upload_to_gcs():
                logger.info("Processing completed successfully")
                return True
            else:
                logger.error("Failed to upload files to GCS")
                return False

        except Exception as e:
            logger.error(f"Error in main process: {e}")
            return False


def main():
    processor = DiamondCatalogProcessor()
    success = processor.run()
    exit(0 if success else 1)


if __name__ == "__main__":
    main()

