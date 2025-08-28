import os
import ftplib
import pandas as pd
import requests
from google.cloud import storage
import random

# ----------------------------
# FTP DOWNLOAD CONFIGURATION
# ----------------------------
FTP_SERVER = "ftp.nivoda.net"
FTP_PORT = 21
FTP_USERNAME = "leeladiamondscorporate@gmail.com"
FTP_PASSWORD = "1yH£lG4n0Mq"

# Use the environment variable FTP_DOWNLOAD_DIR (default to /tmp/raw)
ftp_download_dir = os.environ.get("FTP_DOWNLOAD_DIR", "/tmp/raw")
os.makedirs(ftp_download_dir, exist_ok=True)

ftp_files = {
    "natural": {
         "remote_filename": "Leela Diamond_natural.csv",
         "local_path": os.path.join(ftp_download_dir, "Natural.csv")
    },
    "lab_grown": {
         "remote_filename": "Leela Diamond_labgrown.csv",
         "local_path": os.path.join(ftp_download_dir, "Labgrown.csv")
    },
    "gemstone": {
         "remote_filename": "Leela Diamond_gemstones.csv",
         "local_path": os.path.join(ftp_download_dir, "gemstones.csv")
    }
}

def download_file_from_ftp(remote_filename, local_path):
    """Download a file from the FTP server to a local path."""
    try:
        with ftplib.FTP() as ftp:
            ftp.connect(FTP_SERVER, FTP_PORT)
            ftp.login(FTP_USERNAME, FTP_PASSWORD)
            ftp.set_pasv(True)
            with open(local_path, 'wb') as f:
                ftp.retrbinary(f"RETR {remote_filename}", f.write)
            print(f"Downloaded {remote_filename} to {local_path}")
    except Exception as e:
        print(f"Error downloading {remote_filename}: {e}")

def download_all_files():
    """Download all raw files from the FTP server."""
    for product_type, file_info in ftp_files.items():
        download_file_from_ftp(file_info["remote_filename"], file_info["local_path"])

# ----------------------------
# PINTEREST PROCESSING SCRIPT
# ----------------------------

class DiamondCatalogProcessor:
    def __init__(self):
        # Instead of hard-coded paths, use the downloaded files from FTP.
        # The FTP download step ensures these files are available in ftp_download_dir.
        self.files_to_load = {
            "natural": {
                "file_path": os.path.join(os.environ.get("FTP_DOWNLOAD_DIR", "/tmp/raw"), "Natural.csv")
            },
            "lab_grown": {
                "file_path": os.path.join(os.environ.get("FTP_DOWNLOAD_DIR", "/tmp/raw"), "Labgrown.csv")
            },
            "gemstone": {
                "file_path": os.path.join(os.environ.get("FTP_DOWNLOAD_DIR", "/tmp/raw"), "gemstones.csv")
            }
        }
        # Google Cloud Storage configuration via environment variables
        self.gcs_config = {
            "bucket_name": os.environ.get("BUCKET_NAME", "sitemaps.leeladiamond.com"),
            "bucket_folder": os.environ.get("BUCKET_FOLDER", "pinterestfinal")
        }
        # Output folder for generated files; default to /tmp/pinterest_output
        self.output_folder = os.environ.get("OUTPUT_FOLDER", "/tmp/pinterest_output")
        os.makedirs(self.output_folder, exist_ok=True)

    def process_file(self, file_path, product_type):
        df = pd.read_csv(file_path, dtype=str)
        df = df.fillna('')

        if 'image' in df.columns:
            df['image'] = df['image'].str.extract(r'(https?://.*\.(jpg|png))')[0].fillna('')
            df = df[df['image'] != '']
        else:
            df['image'] = ''

        df['price'] = pd.to_numeric(df.get('price', 0), errors='coerce').fillna(0)

        def markup(x):
            base = x * 1.05 * 1.13
            additional = (
                210 if x <= 500 else
                375 if x <= 1000 else
                500 if x <= 1500 else
                700 if x <= 2000 else
                900 if x <= 2500 else
                1100 if x <= 3000 else
                1200 if x <= 5000 else
                1500 if x <= 100000 else
                0
            ) * 1.15
            return round(base + additional, 2)

        df['price'] = df['price'].apply(markup)
        df['id'] = df['ReportNo'] + "CA"

        product_templates = {
            "natural": {
                "title": lambda row: f"{row['shape']}-{row['carats']} Carats-{row['col']} Color-{row['clar']} Clarity-{row['lab']} Certified-{row['shape']}-Natural Diamond",
                "description": lambda row: f"Discover sustainable luxury with our natural {row['shape']} diamond: {row['carats']} carats, {row['col']} color, and {row['clar']} clarity. Perfect for custom creations. Measurements: {row['length']}-{row['width']}x{row['height']} mm. Cut: {row['cut']}, Polish: {row['pol']}, Symmetry: {row['symm']}, Table: {row['table']}%, Depth: {row['depth']}%, Fluorescence: {row['flo']}. {row['lab']} certified {row['shape']}",
                "link": lambda row: f"https://leeladiamond.com/pages/natural-diamond-catalog?id={row['ReportNo']}"
            },
            "lab_grown": {
                "title": lambda row: f"{row['shape']}-{row['carats']} Carats-{row['col']} Color-{row['clar']} Clarity-{row['lab']} Certified-{row['shape']}-Lab Grown Diamond",
                "description": lambda row: f"Discover sustainable luxury with our lab-grown {row['shape']} diamond: {row['carats']} carats, {row['col']} color, and {row['clar']} clarity. Perfect for custom creations. Measurements: {row['length']}-{row['width']}x{row['height']} mm. Cut: {row['cut']}, Polish: {row['pol']}, Symmetry: {row['symm']}, Table: {row['table']}%, Depth: {row['depth']}%, Fluorescence: {row['flo']}. {row['lab']} certified {row['shape']}",
                "link": lambda row: f"https://leeladiamond.com/pages/lab-grown-diamond-catalog?id={row['ReportNo']}"
            },
            "gemstone": {
                "title": lambda row: f"{row['shape']} {row['Color']} {row['gemType']} Gemstone - {row['carats']} Carats, {row['Clarity']} Clarity, {row['Cut']} Cut, {row['Lab']} Certified",
                "description": lambda row: f"{row['shape']} {row['gemType']} {row['Color']} Gemstone - {row['carats']} carats, color: {row['Color']}, clarity: {row['Clarity']}, cut: {row['Cut']}, lab: {row['Lab']}, treatment: {row['Treatment']}, origin: {row['Mine of Origin']}, size: {row['length']}x{row['width']} mm.",
                "link": lambda row: f"https://leeladiamond.com/pages/gemstone-catalog?id={row['ReportNo']}"
            }
        }

        template = product_templates[product_type]
        df['title'] = df.apply(template['title'], axis=1)
        df['description'] = df.apply(template['description'], axis=1)
        df['link'] = df.apply(template['link'], axis=1)

        num_rows = len(df)
        df['image_link'] = df['image']
        df['availability'] = 'in stock'
        df['google_product_category'] = [random.choice([189, 190, 191, 197, 192, 194, 6463, 196, 200, 5122, 5123, 7471, 6870, 201, 502979, 6540, 6102, 5996, 198, 5982]) for _ in range(num_rows)]
        df['average_review_rating'] = [random.randint(4, 5) for _ in range(num_rows)]
        df['number_of_ratings'] = [random.randint(5, 30) for _ in range(num_rows)]
        df['condition'] = 'new'

        return df[['id', 'title', 'description', 'link', 'image_link', 'price',
                   'availability', 'google_product_category', 'average_review_rating',
                   'number_of_ratings', 'condition']]

    def create_country_specific_files(self, combined_df):
        try:
            response = requests.get("https://v6.exchangerate-api.com/v6/20155ba28afe7c763416cc23/latest/USD")
            if response.status_code != 200:
                raise Exception(f"Failed to fetch exchange rates: {response.status_code}")

            rates = response.json()["conversion_rates"]

            country_currency = {
                "US": "USD", "CA": "CAD", "GB": "GBP", "AU": "AUD",
                "DE": "EUR", "FR": "EUR", "IT": "EUR", "ES": "EUR",
                "NL": "EUR", "SE": "SEK", "CH": "CHF", "JP": "JPY",
                "MX": "MXN", "BR": "BRL", "IN": "INR"
            }

            for country, currency in country_currency.items():
                if currency in rates:
                    country_data = combined_df.copy()
                    country_data['id'] = country_data['id'].str.replace('CA', country)
                    # Ensure price is numeric before conversion
                    country_data['price'] = pd.to_numeric(country_data['price'], errors='coerce').fillna(0)
                    country_data['price'] = (country_data['price'] * rates[currency]).round(2)
                    output_file = os.path.join(self.output_folder, f"{country}-pinterest-csv.csv")
                    country_data.to_csv(output_file, index=False)
                    print(f"Created file for {country}")
        except Exception as e:
            print(f"Error in currency conversion: {e}")

    def upload_to_gcs(self):
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.gcs_config["bucket_name"])

            for file_name in os.listdir(self.output_folder):
                file_path = os.path.join(self.output_folder, file_name)
                if os.path.isfile(file_path):
                    destination_blob_name = f"{self.gcs_config['bucket_folder']}/{file_name}"
                    blob = bucket.blob(destination_blob_name)
                    blob.upload_from_filename(file_path)
                    print(f"Uploaded {file_name} to GCS")
        except Exception as e:
            print(f"GCS upload error: {e}")

    def run(self):
        try:
            # Step 1: Download raw files from FTP
            download_all_files()

            # Step 2: Process each file from the downloaded raw files
            dataframes = []
            for product_type, file_info in self.files_to_load.items():
                dataframes.append(self.process_file(file_info["file_path"], product_type))

            combined_df = pd.concat(dataframes, ignore_index=True)
            # Save the combined catalog in the output folder
            combined_file = os.path.join(self.output_folder, "combined_catalog.csv")
            combined_df.to_csv(combined_file, index=False)
            print(f"Combined catalog saved to {combined_file}")

            # Create country-specific files with updated prices
            self.create_country_specific_files(combined_df)
            # Upload the generated files to Google Cloud Storage
            self.upload_to_gcs()

            print("Processing completed successfully")
        except Exception as e:
            print(f"Error in main process: {e}")

if __name__ == "__main__":
    processor = DiamondCatalogProcessor()
    processor.run()
