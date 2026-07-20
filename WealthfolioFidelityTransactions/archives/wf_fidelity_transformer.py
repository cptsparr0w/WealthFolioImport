import pandas as pd
import numpy as np
import re
import logging
from typing import List, Dict, Optional, Tuple, Pattern
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [FidelityTransformer] - %(message)s'
)
logger = logging.getLogger(__name__)

class FidelityTransformer:
    """
    High-performance transformer for converting Fidelity CSV exports 
    to Wealthfolio-compatible transaction schemas.
    """

    # === Class Constants ===
    
    # Pattern to extract the base ticker from an option string (e.g., FOUR260320P50 -> FOUR)
    OPTION_EXTRACT_PATTERN = re.compile(r'^([A-Z\.\d]+)\d{6}[PC]')

    # Pattern to identify any symbol with an option suffix
    OPTION_SUFFIX_PATTERN = re.compile(r'\d{6}[PC]')

    # Wealthfolio Target Schema: Added 'Currency' column to fix FX Rates error
    TARGET_COLUMNS = ['Date', 'Type', 'Symbol', 'Quantity', 'Price', 'Amount', 'Description', 'Currency']

    # Transaction Type Constants
    TRANSACTION_TYPE_DIVIDEND = 'Dividend'
    
    # Classification Mapping: Precompiled patterns and corresponding Wealthfolio Type
    CLASSIFICATION_MAP: List[Tuple[Pattern[str], str]] = [
        (re.compile(r'TAX|FOREIGN TAX'), 'Tax'),
        (re.compile(r'FEE|COMMISSION'), 'Fee'),
        (re.compile(r'DIVIDEND|REINVESTMENT|CREDIT', re.IGNORECASE), TRANSACTION_TYPE_DIVIDEND),
        (re.compile(r'INTEREST'), 'Interest'),
        (re.compile(r'DEPOSIT|CONTRIBUTION|TRANSFER IN', re.IGNORECASE), 'Deposit'),
        (re.compile(r'WITHDRAWAL|TRANSFER OUT', re.IGNORECASE), 'Withdrawal'),
        (re.compile(r'SPLIT'), 'Stock Split'),
        (re.compile(r'SELL|SOLD'), 'Sell'),
        (re.compile(r'BUY|BOUGHT'), 'Buy'),
    ]

    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean whitespace and standardize casing."""
        cols_to_upper = ['Action', 'Description', 'Symbol']
        for col in cols_to_upper:
            if col in df.columns:
                # astype(str) ensures NaNs are converted to 'nan' string for processing
                df[col] = df[col].astype(str).str.strip().str.upper()
        return df

    def _handle_options_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggressively normalizes symbols to their base tickers, handling both
        raw option symbols and descriptions with embedded tickers.
        
        Strategy: Use a regex to extract the base ticker from the Symbol field,
        regardless of prefixes like " -" or suffixes like "260320P50".
        """
        # Step 1: Strip whitespace and leading dashes/special chars from Symbol
        df['Symbol'] = df['Symbol'].astype(str).str.strip().str.replace(r'^[-\s]+', '', regex=True)
        
        # Step 2: Extract the base ticker from any symbol that contains digits followed by letters
        # Pattern breakdown:
        # - (?<=^[-\s]*): Lookbehind for any leading dashes/spaces (we already stripped them, so this is robust).
        # - ([A-Z\.\d]+): Capture group for the ticker (letters, dots, numbers).
        # - \d{6}[PC]: Match the standard option date pattern and type (P or C).
        #
        # This will turn "FOUR260320P50" and "-FOUR260320P50" into just "FOUR".
        ticker_extraction_pattern = re.compile(r'^([A-Z\.\d]+)\d{6}[PC]')
        
        mask = df['Symbol'].str.contains(r'\d{6}[PC]', na=False)
        
        if mask.any():
            count = mask.sum()
            logger.info(f"Strategy 1: Normalizing {count} option symbol strings to base tickers.")
            
            # Use str.extract with the capture group
            df.loc[mask, 'Symbol'] = df.loc[mask, 'Symbol'].str.extract(ticker_extraction_pattern, expand=False)

        return df



    def _apply_intelligent_categorization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Uses vectorized regex matching with precompiled patterns to assign Wealthfolio transaction types.
        This is significantly faster than inline pattern strings.
        """
        # Create a combined search string for performance
        search_corpus = df['Action'].fillna('') + " " + df['Description'].fillna('')
        
        conditions = []
        choices = []

        # Loop over the compiled patterns for optimal performance
        for pattern, wealthfolio_type in self.CLASSIFICATION_MAP:
            conditions.append(search_corpus.str.contains(pattern, regex=True, na=False))
            choices.append(wealthfolio_type)

        df['Type'] = np.select(conditions, choices, default='Adjustment')
        return df

    def transform(self) -> None:
        """Main Execution Pipeline."""
        # Validate input path early
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        try:
            logger.info(f"Starting transformation for {self.input_path}")
            
            # 1. Read CSV: Skip the first line (Fidelity's "Run Date" metadata) 
            # and let pandas infer the header from the second line.
            df = pd.read_csv(
                str(self.input_path), 
                skiprows=1,           # Skip the first line (e.g., "Run Date,Action,...")
                skipfooter=10,        # Skip the last 10 lines (legal footer)
                engine='python'       # Required for skipfooter
            )

            # 2. CRITICAL FIX: Clean column names to handle any potential whitespace or BOM
            df.columns = df.columns.str.strip()
            
            # 3. Rename 'Run Date' to 'Date' for Wealthfolio compatibility
            df = df.rename(columns={'Run Date': 'Date'})

            # Optional: Debug line to print actual column names found
            logger.info(f"Columns loaded from CSV: {df.columns.tolist()}")

            if df.empty:
                raise ValueError("The input CSV is empty.")

            # 4. Input Validation: Check for critical columns
            required_columns = ['Date', 'Symbol', 'Action']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing critical input columns: {missing_cols}")

            # 5. Data Cleaning
            df = self._preprocess(df)

            # 6. Financial Logic (Option -> Dividend conversion & Symbol normalization)
            df = self._handle_options_logic(df)

            # 7. Intelligent Categorization
            df = self._apply_intelligent_categorization(df)

            # 8. DATE & AMOUNT STANDARDIZATION (Bulletproofing)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d 16:00:00')
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0.0)
            
            # --- CRITICAL FIX: Ensure Price and Quantity are never NaN ---
            df['Price'] = df['Price'].fillna(0.0)
            df['Quantity'] = df['Quantity'].fillna(0.0)
            # ---------------------------------------------------------------

            # --- Currency Fix: Explicitly set currency to USD ---
            logger.info("Setting currency to 'USD' for all transactions.")
            df['Currency'] = 'USD'
            # -------------------------------------------------------

            # --- CLEANUP DIVIDEND SPLITS (KEY FIX) ---
            valid_dividend_mask = ~((df['Type'] == 'Dividend') & (df['Price'].isna()) & (df['Quantity'] == 0.0))
            df = df[valid_dividend_mask].copy()
            logger.info(f"Filtered out invalid dividend rows.")
            
            df.loc[df['Type'] == 'Dividend', 'Price'] = df.loc[df['Type'] == 'Dividend', 'Price'].fillna(1.0)
            # -------------------------------------------

            # 9. Final Schema Alignment
            logger.info("Aligning final schema.")
            
            df['Symbol'] = df['Symbol'].replace('NAN', 'CASH').fillna('CASH')
            
            # --- FINAL DATA CLEANUP: Drop rows with critical nulls ---
            final_df = df[self.TARGET_COLUMNS].dropna(subset=['Date', 'Type', 'Amount']).copy()
            logger.info(f"After dropping invalid rows: {len(final_df)} entries remaining.")
            # ------------------------------------------------------------

            # 10. Final Validation
            self._validate(final_df)

            # Export: convert str Path to string
            final_df.to_csv(str(self.output_path), index=False)
            logger.info(f"Success! Processed file saved to: {self.output_path}")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    def _validate(self, df: pd.DataFrame) -> None:
        """Final integrity check before writing to disk."""
        critical_cols = ['Date', 'Type', 'Amount']
        null_counts = df[critical_cols].isna().sum()
        if null_counts.any():
            logger.warning(f"Integrity Warning: Nulls detected in critical columns:\n{null_counts}")


if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "fidelity_export.csv"
    OUTPUT_FILE = "wealthfolio_import.csv"

    transformer = FidelityTransformer(INPUT_FILE, OUTPUT_FILE)
    transformer.transform()
