import os
import ftplib
import pandas as pd
import requests
from google.cloud import storage
import random
import logging
from typing import Dict, Optional
import time
from decimal import Decimal, ROUND_HALF_UP
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DiamondCatalogProcessor:
    """Enhanced Diamond Catalog Processor with improved accuracy and error handling."""

    def __init__(self):
        # FTP Configuration - Use environment variables for security
        self.ftp_config = {
            "server": os.environ.get("FTP_SERVER", "ftp.nivoda.net"),
            "port": int(os.environ.get("FTP_PORT", "21")),
            "username": os.environ.get("FTP_USERNAME", "leeladiamondscorporate@gmail.com"),
            "password": os.environ.get("FTP_PASSWORD", "1yH£lG4n0Mq"),
        }

        # Directories
        self.ftp_download_dir = os.environ.get("FTP_DOWNLOAD_DIR", "/tmp/raw")
        self.output_folder = os.environ.get("OUTPUT_FOLDER", "/tmp/pinterest_output")

        os.makedirs(self.ftp_download_dir, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)

        # FTP file mappings
        self.ftp_files = {
            "natural": {
                "remote_filename": "Leela Diamond_natural.csv",
                "local_path": os.path.join(self.ftp_download_dir, "Natural.csv"),
            },
            "lab_grown": {
                "remote_filename": "Leela Diamond_labgrown.csv",
                "local_path": os.path.join(self.ftp_download_dir, "Labgrown.csv"),
            },
            "gemstone": {
                "remote_filename": "Leela Diamond_gemstones.csv",
                "local_path": os.path.join(self.ftp_download_dir, "gemstones.csv"),
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
        self.jewelry_categories = [
            189, 190, 191, 197, 192, 194, 6463, 196, 200,
            5122, 5123, 7471, 6870, 201, 502979, 6540,
            6102, 5996, 198, 5982
        ]

    # ----------------------------
    # FTP
    # ----------------------------
    def download_file_from_ftp(self, remote_filename: str, local_path: str, max_retries: int = 3) -> bool:
        for attempt in range(max_retries):
            try:
                with ftplib.FTP() as ftp:
                    ftp.connect(self.ftp_config["server"], self.ftp_config["port"], timeout=30)
                    ftp.login(self.ftp_config["username"], self.ftp_config["password"])
                    ftp.set_pasv(True)
                    with open(local_path, 'wb') as f:
                        ftp.retrbinary(f"RETR {remote_filename}", f.write)
                logger.info(f"Successfully downloaded {remote_filename} to {local_path}")
                return True
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed downloading {remote_filename}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed to download {remote_filename} after {max_retries} attempts")
        return False

    def download_all_files(self) -> bool:
        logger.info("Starting FTP download process...")
        success_count = 0
        for product_type, file_info in self.ftp_files.items():
            if self.download_file_from_ftp(file_info["remote_filename"], file_info["local_path"]):
                success_count += 1
            else:
                logger.error(f"Failed to download {product_type} file")
        logger.info(f"Downloaded {success_count}/{len(self.ftp_files)} files successfully")
        return success_count == len(self.ftp_files)

    # ----------------------------
    # Pricing / Currency
    # ----------------------------
    def convert_to_cad(self, price_usd):
        """Convert price from USD to CAD using a fixed exchange rate."""
        cad_rate = 1.41
        try:
            return round(float(price_usd) * cad_rate, 2)
        except (ValueError, TypeError) as e:
            logger.warning(f"Currency conversion skipped for invalid value {price_usd}: {e}")
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

    @staticmethod
    def _parse_price_number(value: str) -> float:
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

    # ----------------------------
    # Cleaning / Validation
    # ----------------------------
    def clean_and_validate_data(self, df: pd.DataFrame, product_type: str) -> pd.DataFrame:
        logger.info(f"Cleaning {product_type} data - {len(df)} rows")
        df = df.fillna('')

        # Require valid image for Pinterest
        if 'image' in df.columns:
            df['image'] = df['image'].str.extract(r'(https?://[^\s]+\.(?:jpg|jpeg|png|webp))', expand=False).fillna('')
            before = len(df)
            df = df[df['image'].str.len() > 0]
            logger.info(f"Filtered {before - len(df)} rows without valid images")

        # Do NOT filter by original 'price'—we compute price from markupPrice later.
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
    # Processing
    # ----------------------------
    def process_file(self, file_path: str, product_type: str) -> pd.DataFrame:
        try:
            logger.info(f"Processing {product_type} file: {file_path}")

            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                logger.error(f"Missing or empty file: {file_path}")
                return pd.DataFrame()

            try:
                df = pd.read_csv(file_path, dtype=str, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, dtype=str, encoding='latin-1')

            if df.empty:
                logger.warning(f"No data in {product_type} file")
                return pd.DataFrame()

            df = self.clean_and_validate_data(df, product_type)
            if df.empty:
                logger.warning(f"No valid data remaining in {product_type} file after cleaning")
                return pd.DataFrame()

            # ---- PRICE PIPELINE (MANDATORY) ----
            # Use markupPrice (USD). Convert to CAD. Drop rows with no usable price.
            if 'markupPrice' not in df.columns:
                df['markupPrice'] = 0

            df['markupPrice'] = pd.to_numeric(df['markupPrice'], errors='coerce').fillna(0.0)
            df['price_cad'] = df['markupPrice'].apply(self.convert_to_cad)

            # Drop rows with zero/invalid price to satisfy “Enter price values…” constraint
            before_pr = len(df)
            df = df[df['price_cad'] > 0]
            dropped = before_pr - len(df)
            if dropped:
                logger.info(f"Dropped {dropped} rows without valid price")

            # Format for Merchant / Pinterest: '123.45 CAD'
            df['price'] = df['price_cad'].apply(lambda x: self.format_price(x, 'CAD'))
            df['markupCurrency'] = 'CAD'

            # Generate unique IDs
            df['id'] = df.get('ReportNo', '').astype(str) + "CA"

            # Product info
            df = self.apply_product_templates(df, product_type)

            # Pinterest fields
            df = self.add_pinterest_fields(df)

            # Final safety: ensure no empty price strings remain
            df = df[df['price'].astype(str).str.match(r'^\d+\.\d{2}\s[A-Z]{3}$', na=False)]

            logger.info(f"Successfully processed {len(df)} {product_type} products")
            return df

        except Exception as e:
            logger.error(f"Error processing {product_type} file {file_path}: {e}")
            return pd.DataFrame()

    def apply_product_templates(self, df: pd.DataFrame, product_type: str) -> pd.DataFrame:
        product_templates = {
            "natural": {
                "title": lambda row: f"{row.get('shape', 'DIAMOND')} {row.get('carats', '')} Carats {row.get('col', '')} Color {row.get('clar', '')} Clarity {row.get('lab', '')} Certified Natural Diamond",
                "description": lambda row: (
                    f"Discover sustainable luxury with our natural {row.get('shape', 'diamond')}: "
                    f"{row.get('carats', '')} carats, {row.get('col', '')} color, and {row.get('clar', '')} clarity. "
                    f"Measurements: {row.get('length', '')}-{row.get('width', '')}x{row.get('height', '')} mm. "
                    f"Cut: {row.get('cut', '')}, Polish: {row.get('pol', '')}, Symmetry: {row.get('symm', '')}, "
                    f"Table: {row.get('table', '')}%, Depth: {row.get('depth', '')}%, Fluorescence: {row.get('flo', '')}. "
                    f"{row.get('lab', '')} certified natural diamond."
                ),
                "link": lambda row: f"https://leeladiamond.com/pages/natural-diamond-catalog?id={row.get('ReportNo', '')}",
            },
            "lab_grown": {
                "title": lambda row: (
                    f"{row.get('shape','')} {row.get('carats','')} Carats "
                    f"{row.get('col','')} Color {row.get('clar','')} Clarity "
                    f"{row.get('lab','')} Certified Lab Grown Diamond"
                ),
                "description": lambda row: (
                    f"Discover sustainable luxury with our lab-grown {row.get('shape','')} diamond: "
                    f"{row.get('carats','')} carats, {row.get('col','')} color, and {row.get('clar','')} clarity. "
                    f"Measurements: {row.get('length','')}-{row.get('width','')}x{row.get('height','')} mm. "
                    f"Cut: {row.get('cut','')}, Polish: {row.get('pol','')}, Symmetry: {row.get('symm','')}, "
                    f"Table: {row.get('table','')}%, Depth: {row.get('depth','')}%, Fluorescence: {row.get('flo','')}. "
                    f"{row.get('lab','')} certified {row.get('shape','')}"
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
                    f"{row.get('shape', '')} {row.get('Color', '')} {row.get('gemType', 'Gemstone')} - "
                    f"{row.get('carats', '')} Carats {row.get('Clarity', '')} Clarity {row.get('Lab', '')} Certified"
                ),
                "description": lambda row: (
                    f"Beautiful {row.get('shape', '')} {row.get('gemType', 'gemstone')} in {row.get('Color', '')} - "
                    f"{row.get('carats', '')} carats with {row.get('Clarity', '')} clarity. "
                    f"Cut: {row.get('Cut', '')}, Lab: {row.get('Lab', '')}, Treatment: {row.get('Treatment', '')}, "
                    f"Origin: {row.get('Mine of Origin', '')}, Size: {row.get('length', '')}x{row.get('width', '')} mm."
                ),
                "link": lambda row: f"https://leeladiamond.com/pages/gemstone-catalog?id={row.get('ReportNo', '')}",
            },
        }

        if product_type in product_templates:
            template = product_templates[product_type]
            df['title'] = df.apply(template['title'], axis=1)
            df['description'] = df.apply(template['description'], axis=1)
            df['link'] = df.apply(template['link'], axis=1)

        df['title'] = df['title'].str.replace(r'\s+', ' ', regex=True).str.strip()
        df['description'] = df['description'].str.replace(r'\s+', ' ', regex=True).str.strip()
        return df

    # ----------------------------
    # Pinterest fields
    # ----------------------------
    def add_pinterest_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        num_rows = len(df)
        df['image_link'] = df['image']
        df['availability'] = 'in stock'
        df['google_product_category'] = [random.choice(self.jewelry_categories) for _ in range(num_rows)]
        df['average_review_rating'] = [round(random.uniform(4.0, 5.0), 1) for _ in range(num_rows)]
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
                        logger.error(f"Exchange rate API error: {data.get('error-type', 'Unknown error')}")
                else:
                    logger.error(f"HTTP error fetching exchange rates: {response.status_code}")
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

    def create_country_specific_files(self, combined_df: pd.DataFrame) -> None:
        logger.info("Creating country-specific files...")
        rates = self.get_exchange_rates()
        if not rates:
            logger.error("No exchange rates available, skipping country-specific file creation")
            return

        usd_to_cad = Decimal(str(rates.get("CAD", 1))) or Decimal("1")
        files_created = 0

        for country, currency in self.country_currency.items():
            try:
                if currency not in rates:
                    logger.warning(f"Exchange rate not available for {currency}, skipping {country}")
                    continue

                country_data = combined_df.copy()
                country_data['id'] = country_data['id'].str.replace(r'CA$', country, regex=True)

                target_rate = Decimal(str(rates[currency]))
                cad_to_target = (target_rate / usd_to_cad)

                country_data['price_numeric'] = country_data['price'].apply(self._parse_price_number)
                country_data['price_numeric'] = country_data['price_numeric'].apply(
                    lambda x: float((Decimal(str(x)) * cad_to_target).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
                )
                country_data['price'] = country_data['price_numeric'].map(lambda x: self.format_price(x, currency))
                country_data.drop(columns=['price_numeric'], inplace=True, errors='ignore')

                output_file = os.path.join(self.output_folder, f"{country}-pinterest-csv.csv")
                # Final guard: keep only rows with valid price format
                country_data = country_data[country_data['price'].astype(str).str.match(r'^\d+\.\d{2}\s[A-Z]{3}$', na=False)]
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
                        destination_blob_name = f"{self.gcs_config['bucket_folder'].rstrip('/')}/{file_name}"
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
                logger.error("Failed to download required files from FTP")
                return False

            dataframes = []
            for product_type, file_info in self.ftp_files.items():
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
            combined_df = combined_df[combined_df['price'].astype(str).str.match(r'^\d+\.\d{2}\s[A-Z]{3}$', na=False)]
            after_final = len(combined_df)
            if before_final != after_final:
                logger.info(f"Removed {before_final - after_final} rows lacking a valid formatted price")

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
