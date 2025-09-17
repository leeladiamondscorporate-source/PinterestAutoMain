import os
import ftplib
import pandas as pd
import requests
from google.cloud import storage
import random
import logging
from typing import Dict, List, Optional
import time
from decimal import Decimal, ROUND_HALF_UP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DiamondCatalogProcessor:
    """Enhanced Diamond Catalog Processor with improved accuracy and error handling."""
    
    def __init__(self):
        """Initialize the processor with configuration from environment variables."""
        # FTP Configuration - Use environment variables for security
        self.ftp_config = {
            "server": os.environ.get("FTP_SERVER", "ftp.nivoda.net"),
            "port": int(os.environ.get("FTP_PORT", "21")),
            "username": os.environ.get("FTP_USERNAME", "leeladiamondscorporate@gmail.com"),
            "password": os.environ.get("FTP_PASSWORD", "1yH£lG4n0Mq")
        }
        
        # Directories
        self.ftp_download_dir = os.environ.get("FTP_DOWNLOAD_DIR", "/tmp/raw")
        self.output_folder = os.environ.get("OUTPUT_FOLDER", "/tmp/pinterest_output")
        
        # Create directories
        os.makedirs(self.ftp_download_dir, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        
        # FTP file mappings
        self.ftp_files = {
            "natural": {
                "remote_filename": "Leela Diamond_natural.csv",
                "local_path": os.path.join(self.ftp_download_dir, "Natural.csv")
            },
            "lab_grown": {
                "remote_filename": "Leela Diamond_labgrown.csv",
                "local_path": os.path.join(self.ftp_download_dir, "Labgrown.csv")
            },
            "gemstone": {
                "remote_filename": "Leela Diamond_gemstones.csv",
                "local_path": os.path.join(self.ftp_download_dir, "gemstones.csv")
            }
        }
        
        # Google Cloud Storage configuration
        self.gcs_config = {
            "bucket_name": os.environ.get("BUCKET_NAME", "sitemaps.leeladiamond.com"),
            "bucket_folder": os.environ.get("BUCKET_FOLDER", "pinterestfinal")
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
        self.jewelry_categories = [189, 190, 191, 197, 192, 194, 6463, 196, 200, 
                                 5122, 5123, 7471, 6870, 201, 502979, 6540, 
                                 6102, 5996, 198, 5982]

    def download_file_from_ftp(self, remote_filename: str, local_path: str, max_retries: int = 3) -> bool:
        """Download a file from FTP server with retry logic."""
        for attempt in range(max_retries):
            try:
                with ftplib.FTP() as ftp:
                    ftp.connect(self.ftp_config["server"], self.ftp_config["port"])
                    ftp.login(self.ftp_config["username"], self.ftp_config["password"])
                    ftp.set_pasv(True)
                    
                    with open(local_path, 'wb') as f:
                        ftp.retrbinary(f"RETR {remote_filename}", f.write)
                    
                    logger.info(f"Successfully downloaded {remote_filename} to {local_path}")
                    return True
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed downloading {remote_filename}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to download {remote_filename} after {max_retries} attempts")
        return False

    def download_all_files(self) -> bool:
        """Download all required files from FTP server."""
        logger.info("Starting FTP download process...")
        success_count = 0
        
        for product_type, file_info in self.ftp_files.items():
            if self.download_file_from_ftp(file_info["remote_filename"], file_info["local_path"]):
                success_count += 1
            else:
                logger.error(f"Failed to download {product_type} file")
        
        logger.info(f"Downloaded {success_count}/{len(self.ftp_files)} files successfully")
        return success_count == len(self.ftp_files)

    def convert_to_cad(price_usd):
    """Convert price from USD to CAD using a fixed exchange rate."""
    cad_rate = 1.41  # set via ENV or config if needed
    try:
        price_usd = float(price_usd)  # ensure numeric
        return round(price_usd * cad_rate, 2)
    except (ValueError, TypeError) as e:
        print(f"[WARN] Currency conversion skipped for invalid value {price_usd}: {e}")
        return 0.0  # safer fallback


    def clean_and_validate_data(self, df: pd.DataFrame, product_type: str) -> pd.DataFrame:
        """Clean and validate the data with improved accuracy."""
        logger.info(f"Cleaning {product_type} data - {len(df)} rows")
        
        # Fill NaN values
        df = df.fillna('')
        
        # Clean image URLs with improved regex
        if 'image' in df.columns:
            # Extract valid image URLs (jpg, png, webp, jpeg)
            df['image'] = df['image'].str.extract(r'(https?://[^\s]+\.(jpg|jpeg|png|webp))', expand=False)[0]
            df['image'] = df['image'].fillna('')
            
            # Filter out rows without valid images for Pinterest
            initial_count = len(df)
            df = df[df['image'].str.len() > 0]
            logger.info(f"Filtered {initial_count - len(df)} rows without valid images")
        
        # Clean and validate price
        df['price'] = pd.to_numeric(df.get('price', 0), errors='coerce').fillna(0)
        
        # Filter out zero or negative prices
        df = df[df['price'] > 0]
        
        # Clean text fields
        text_columns = ['shape', 'col', 'clar', 'cut', 'pol', 'symm', 'lab']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
        
        # Validate required fields based on product type
        required_fields = {
            'natural': ['ReportNo', 'shape', 'carats', 'col', 'clar', 'lab'],
            'lab_grown': ['ReportNo', 'shape', 'carats', 'col', 'clar', 'lab'],
            'gemstone': ['ReportNo', 'shape', 'carats', 'Color', 'gemType']
        }
        
        if product_type in required_fields:
            for field in required_fields[product_type]:
                if field in df.columns:
                    df = df[df[field].astype(str).str.len() > 0]
        
        logger.info(f"After cleaning: {len(df)} rows remaining")
        return df

    def process_file(self, file_path: str, product_type: str) -> pd.DataFrame:
        """Process individual file with enhanced data validation."""
        try:
            logger.info(f"Processing {product_type} file: {file_path}")
            
            # Check if file exists and is not empty
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return pd.DataFrame()
            
            if os.path.getsize(file_path) == 0:
                logger.error(f"File is empty: {file_path}")
                return pd.DataFrame()
            
            # Read CSV with error handling
            try:
                df = pd.read_csv(file_path, dtype=str, encoding='utf-8')
            except UnicodeDecodeError:
                # Try with different encoding
                df = pd.read_csv(file_path, dtype=str, encoding='latin-1')
            
            if df.empty:
                logger.warning(f"No data in {product_type} file")
                return pd.DataFrame()
            
            # Clean and validate data
            df = self.clean_and_validate_data(df, product_type)
            
            if df.empty:
                logger.warning(f"No valid data remaining in {product_type} file after cleaning")
                return pd.DataFrame()
            
           # ---- PRICE: convert existing markupPrice (assumed USD) -> CAD; show on 'price' ----
            if 'markupPrice' in df.columns:
                df['markupPrice'] = pd.to_numeric(df['markupPrice'], errors='coerce')
            else:
                # create a zero series if the column doesn't exist
                df['markupPrice'] = pd.Series(0.0, index=df.index)
            
            df['markupPrice'] = df['markupPrice'].fillna(0.0).astype(float)
            df['markupPrice'] = df['markupPrice'].apply(convert_to_cad)
            
            # Google Merchant Center price format
            df['price'] = df['markupPrice'].map(lambda x: f"{x:.2f} CAD")
            df['markupCurrency'] = 'CAD'
            
            # Generate unique IDs
            df['id'] = df['ReportNo'].astype(str) + "CA"
            
            # Generate product information using templates
            df = self.apply_product_templates(df, product_type)
            
            # Add Pinterest-specific fields
            df = self.add_pinterest_fields(df)
            
            logger.info(f"Successfully processed {len(df)} {product_type} products")
            return df
            
        except Exception as e:
            logger.error(f"Error processing {product_type} file {file_path}: {e}")
            return pd.DataFrame()

    def apply_product_templates(self, df: pd.DataFrame, product_type: str) -> pd.DataFrame:
        """Apply product-specific templates for title, description, and link generation."""
        product_templates = {
            "natural": {
                "title": lambda row: f"{row.get('shape', 'Diamond')} {row.get('carats', '')} Carats {row.get('col', '')} Color {row.get('clar', '')} Clarity {row.get('lab', '')} Certified Natural Diamond",
                "description": lambda row: f"Discover sustainable luxury with our natural {row.get('shape', 'diamond')}: {row.get('carats', '')} carats, {row.get('col', '')} color, and {row.get('clar', '')} clarity. Perfect for custom jewelry creations. Measurements: {row.get('length', '')}-{row.get('width', '')}x{row.get('height', '')} mm. Cut: {row.get('cut', '')}, Polish: {row.get('pol', '')}, Symmetry: {row.get('symm', '')}, Table: {row.get('table', '')}%, Depth: {row.get('depth', '')}%, Fluorescence: {row.get('flo', '')}. {row.get('lab', '')} certified natural diamond.",
                "link": lambda row: f"https://leeladiamond.com/pages/natural-diamond-catalog?id={row.get('ReportNo', '')}"
            },
             "lab_grown": {
                    "title": lambda row: f"{row.get('shape','')}-{row.get('carats','')} Carats-{row.get('col','')} Color-{row.get('clar','')} Clarity-{row.get('lab','')} Certified-{row.get('shape','')}-Lab Grown Diamond",
                    "description": lambda row: (
                        f"Discover sustainable luxury with our lab-grown {row.get('shape','')} diamond: "
                        f"{row.get('carats','')} carats, {row.get('col','')} color, and {row.get('clar','')} clarity. "
                        f"Measurements: {row.get('length','')}-{row.get('width','')}x{row.get('height','')} mm. "
                        f"Cut: {row.get('cut','')}, Polish: {row.get('pol','')}, Symmetry: {row.get('symm','')}, "
                        f"Table: {row.get('table','')}%, Depth: {row.get('depth','')}%, Fluorescence: {row.get('flo','')}. "
                        f"{row.get('lab','')} certified {row.get('shape','')}"
                    ),
                    # SEO-friendly link (carats 1.50 -> 1-50, lowercased, hyphenated)
                    "link": lambda row: (
                        "https://leeladiamond.com/pages/lab-grown-diamonds/"
                        f"{str(row.get('shape','')).strip().lower()}-"
                        f"{str(row.get('carats','')).replace('.', '-')}-carat-"
                        f"{str(row.get('col','')).strip().lower()}-color-"
                        f"{str(row.get('clar','')).strip().lower()}-clarity-"
                        f"{str(row.get('lab','')).strip().lower()}-certified-"
                        f"{str(row.get('ReportNo','')).strip()}"
                    )
            },
            "gemstone": {
                "title": lambda row: f"{row.get('shape', '')} {row.get('Color', '')} {row.get('gemType', 'Gemstone')} - {row.get('carats', '')} Carats {row.get('Clarity', '')} Clarity {row.get('Lab', '')} Certified",
                "description": lambda row: f"Beautiful {row.get('shape', '')} {row.get('gemType', 'gemstone')} in {row.get('Color', '')} - {row.get('carats', '')} carats with {row.get('Clarity', '')} clarity. Cut: {row.get('Cut', '')}, Lab: {row.get('Lab', '')}, Treatment: {row.get('Treatment', '')}, Origin: {row.get('Mine of Origin', '')}, Size: {row.get('length', '')}x{row.get('width', '')} mm. Perfect for custom jewelry designs.",
                "link": lambda row: f"https://leeladiamond.com/pages/gemstone-catalog?id={row.get('ReportNo', '')}"
            }
        }
        
        if product_type in product_templates:
            template = product_templates[product_type]
            df['title'] = df.apply(template['title'], axis=1)
            df['description'] = df.apply(template['description'], axis=1)
            df['link'] = df.apply(template['link'], axis=1)
        
        # Clean up templates (remove extra spaces, empty fields)
        df['title'] = df['title'].str.replace(r'\s+', ' ', regex=True).str.strip()
        df['description'] = df['description'].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        return df

    def add_pinterest_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Pinterest-specific fields with realistic values."""
        num_rows = len(df)
        
        df['image_link'] = df['image']
        df['availability'] = 'in stock'
        
        # More realistic category distribution
        df['google_product_category'] = [random.choice(self.jewelry_categories) for _ in range(num_rows)]
        
        # More realistic review ratings and counts
        df['average_review_rating'] = [round(random.uniform(4.0, 5.0), 1) for _ in range(num_rows)]
        df['number_of_ratings'] = [random.randint(5, 50) for _ in range(num_rows)]
        df['condition'] = 'new'
        
        return df

    def get_exchange_rates(self) -> Optional[Dict]:
        """Get current exchange rates with error handling and retries."""
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
                time.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error("Failed to fetch exchange rates, using fallback rates")
        # Fallback exchange rates (update these periodically)
        return {
            "USD": 1.0, "CAD": 1.35, "GBP": 0.79, "EUR": 0.85, "AUD": 1.50,
            "JPY": 110.0, "CHF": 0.92, "SEK": 8.5, "MXN": 20.0, "BRL": 5.2,
            "INR": 74.0, "SGD": 1.35, "HKD": 7.8, "NZD": 1.60, "DKK": 6.3, "NOK": 8.8
        }

    def create_country_specific_files(self, combined_df: pd.DataFrame) -> None:
        """Create country-specific files with accurate currency conversion."""
        logger.info("Creating country-specific files...")
        
        # Get current exchange rates
        rates = self.get_exchange_rates()
        if not rates:
            logger.error("No exchange rates available, skipping country-specific file creation")
            return
        
        files_created = 0
        
        for country, currency in self.country_currency.items():
            try:
                if currency not in rates:
                    logger.warning(f"Exchange rate not available for {currency}, skipping {country}")
                    continue
                
                country_data = combined_df.copy()
                
                # Update IDs for the country
                country_data['id'] = country_data['id'].str.replace('CA$', country, regex=False)
                
                # Convert prices with proper rounding
                rate = Decimal(str(rates[currency]))
                country_data['price'] = country_data['price'].apply(
                    lambda x: float(Decimal(str(x)) * rate)
                ).round(2)
                
                # Format price with currency
                country_data['price'] = country_data['price'].astype(str) + f" {currency}"
                
                # Save country-specific file
                output_file = os.path.join(self.output_folder, f"{country}-pinterest-csv.csv")
                country_data.to_csv(output_file, index=False, encoding='utf-8')
                
                logger.info(f"Created {country} file with {len(country_data)} products")
                files_created += 1
                
            except Exception as e:
                logger.error(f"Error creating file for {country}: {e}")
        
        logger.info(f"Created {files_created} country-specific files")

    def upload_to_gcs(self) -> bool:
        """Upload files to Google Cloud Storage with error handling."""
        try:
            logger.info("Starting GCS upload...")
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.gcs_config["bucket_name"])
            
            files_uploaded = 0
            
            for file_name in os.listdir(self.output_folder):
                file_path = os.path.join(self.output_folder, file_name)
                
                if os.path.isfile(file_path):
                    try:
                        destination_blob_name = f"{self.gcs_config['bucket_folder']}/{file_name}"
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

    def run(self) -> bool:
        """Main execution method with comprehensive error handling."""
        try:
            logger.info("Starting Diamond Catalog Processing...")
            
            # Step 1: Download files from FTP
            if not self.download_all_files():
                logger.error("Failed to download required files from FTP")
                return False
            
            # Step 2: Process each product type
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
            
            # Step 3: Combine all dataframes
            combined_df = pd.concat(dataframes, ignore_index=True)
            
            # Select final columns for output
            final_columns = ['id', 'title', 'description', 'link', 'image_link', 'price',
                           'availability', 'google_product_category', 'average_review_rating',
                           'number_of_ratings', 'condition']
            
            # Ensure all required columns exist
            for col in final_columns:
                if col not in combined_df.columns:
                    combined_df[col] = ''
            
            combined_df = combined_df[final_columns]
            
            # Save combined catalog
            combined_file = os.path.join(self.output_folder, "combined_catalog.csv")
            combined_df.to_csv(combined_file, index=False, encoding='utf-8')
            logger.info(f"Combined catalog saved with {len(combined_df)} total products")
            
            # Step 4: Create country-specific files
            self.create_country_specific_files(combined_df)
            
            # Step 5: Upload to Google Cloud Storage
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
    """Main entry point."""
    processor = DiamondCatalogProcessor()
    success = processor.run()
    exit(0 if success else 1)

if __name__ == "__main__":
    main()

