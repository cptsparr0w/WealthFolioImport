import pandas as pd
import numpy as np
import re
import logging
from typing import List, Dict, Optional, Tuple, Pattern
from pathlib import Path
import os

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
    OPTION_EXTRACT_PATTERN = re.compile(r'^([A-Z\.\d]+)\d{6}[PC]')
    OPTION_SUFFIX_PATTERN = re.compile(r'\d{6}[PC]')
    TARGET_COLUMNS = ['Date', 'Type', 'Symbol', 'Quantity', 'Price', 'Amount', 'Description', 'Currency']

    TRANSACTION_TYPE_DIVIDEND = 'Dividend'
    
    # Classification Mapping for Wealthfolio
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

    def __init__(self, input_path: str = None, output_path: str = None):
        # Determine paths automatically if not provided
        current_dir = Path(os.getcwd())
        
        if input_path is None:
            # Look for fidelity CSV files in current directory
            csv_files = list(current_dir.glob("History_for_Account_*.csv"))
            if not csv_files:
                # Fall back to a default filename
                self.input_path = current_dir / "fidelity_export.csv"
            else:
                # Use the first CSV file found
                self.input_path = csv_files[0]
                logger.info(f"Found input file: {self.input_path}")
        else:
            self.input_path = Path(input_path)
            
        if output_path is None:
            # Create output filename based on input
            stem = self.input_path.stem
            self.output_path = current_dir / f"wealthfolio_import_{stem}.csv"
        else:
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
        conditions, choices = [], []

        # Loop over the compiled patterns for optimal performance
        for pattern, wealthfolio_type in self.CLASSIFICATION_MAP:
            conditions.append(search_corpus.str.contains(pattern, regex=True, na=False))
            choices.append(wealthfolio_type)

        df['Type'] = np.select(conditions, choices, default='Adjustment')
        return df

    def _handle_adjustments_as_deposits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forces 'Adjustment' (CASH) to 'Deposit'.
        This is the KEY FIX for "Account has negative portfolio balance".
        Wealthfolio only recognizes DEPOSIT for external funds.
        """
        # Identify rows to convert: Symbol must be 'CASH' or empty, Type must be 'Adjustment'
        mask = ((df['Symbol'] == 'CASH') | (df['Symbol'].isna() | (df['Symbol'] == ''))) & (df['Type'] == 'Adjustment')
        
        if mask.any():
            earliest_date = df['Date'].min()
            count = mask.sum()
            logger.info(f"Converting {count} 'Adjustment' (CASH) to 'Deposit'. Date: {earliest_date}.")
            df.loc[mask, 'Type'] = 'Deposit'
            
            # Pushing to earliest date prevents any negative balance at the start.
            df.loc[mask, 'Date'] = earliest_date
            
        return df

    def _handle_missing_cost_basis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort by Date to ensure buys are processed before sells.
        Chronological order is crucial for Wealthfolio to match cost-basis correctly.
        """
        logger.info("Sorting transactions chronologically to resolve potential cost-basis mismatches.")
        return df.sort_values(by='Date')

    def transform(self) -> None:
        """Main Execution Pipeline."""
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        try:
            logger.info(f"Starting transformation for {self.input_path}")
            
            # 1. Read CSV: Skip the first line (Fidelity's "Run Date" metadata)
            df = pd.read_csv(str(self.input_path), skiprows=1, skipfooter=10, engine='python')
            df.columns = df.columns.str.strip()
            
            # 2. Rename 'Run Date' to 'Date'
            if 'Run Date' in df.columns:
                df = df.rename(columns={'Run Date': 'Date'})

            if df.empty:
                raise ValueError("The input CSV is empty.")

            # 3. Input Validation
            required_columns = ['Date', 'Symbol', 'Action']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing critical input columns: {missing_cols}")

            # 4. Data Cleaning & Processing
            df = self._preprocess(df)
            df = self._handle_options_logic(df)
            df = self._apply_intelligent_categorization(df)

            # 5. DATE & AMOUNT STANDARDIZATION
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d 16:00:00')
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0.0)
            df['Price'] = df['Price'].fillna(0.0)
            df['Quantity'] = df['Quantity'].fillna(0.0)

            # --- Currency Fix: Explicitly set currency to USD ---
            logger.info("Setting currency to 'USD' for all transactions.")
            df['Currency'] = 'USD'

            # --- CLEANUP DIVIDEND SPLITS (KEY FIX) ---
            valid_dividend_mask = ~((df['Type'] == 'Dividend') & (df['Price'].isna()) & (df['Quantity'] == 0.0))
            df = df[valid_dividend_mask].copy()
            logger.info(f"Filtered out invalid dividend rows. Remaining: {len(df)}")

            # === THE CRITICAL FIXES FOR BOTH WARNINGS ===
            # 6. Convert 'Adjustment' (CASH) to 'Deposit' (Fixes Negative Balance)
            df = self._handle_adjustments_as_deposits(df)

            # 7. Chronological Sort (Fixes Cost-Basis Mismatch)
            logger.info("Sorting transactions chronologically to resolve potential cost-basis mismatches.")
            df = self._handle_missing_cost_basis(df)

            # --- FINAL Schema & Cleanup ---
            logger.info("Aligning final schema.")
            df['Symbol'] = df['Symbol'].replace('NAN', 'CASH').fillna('CASH')
            final_df = df[self.TARGET_COLUMNS].dropna(subset=['Date', 'Type', 'Amount']).copy()
            logger.info(f"After dropping invalid rows: {len(final_df)} entries remaining.")

            # Export
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


def transform_all_fidelity_files(input_dir: str = None, output_prefix: str = "wealthfolio_import"):
    """
    Process all Fidelity CSV files in a directory.
    
    Args:
        input_dir: Directory containing the CSV files. If None, uses current working directory.
        output_prefix: Prefix for output filenames
    """
    # Determine input directory
    if input_dir is None:
        input_dir = os.getcwd()
    
    input_path = Path(input_dir)
    csv_files = list(input_path.glob("History_for_Account_*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {input_dir}")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    # Process each file
    for csv_file in csv_files:
        logger.info(f"Processing: {csv_file}")
        
        try:
            # Generate output filename based on input
            stem = csv_file.stem
            output_path = input_path / f"{output_prefix}_{stem}.csv"
            
            # Create transformer and process the file
            transformer = FidelityTransformer(str(csv_file), str(output_path))
            transformer.transform()
            
        except Exception as e:
            logger.error(f"Failed to process {csv_file}: {e}", exc_info=True)


if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            # Process all CSV files
            transform_all_fidelity_files()
        else:
            # Use specified input file
            transformer = FidelityTransformer(sys.argv[1])
            transformer.transform()
    else:
        # Default: use current directory, look for specific file or process all
        import glob
        
        # Look for files in the current directory
        csv_files = glob.glob("History_for_Account_*.csv")
        
        if len(csv_files) > 1:
            logger.info(f"Found {len(csv_files)} CSV files. Processing all...")
            transform_all_fidelity_files()
        elif len(csv_files) == 1:
            logger.info(f"Found single CSV file: {csv_files[0]}")
            transformer = FidelityTransformer(csv_files[0])
            transformer.transform()
        else:
            # Fall back to default filename
            logger.info("No History_for_Account_*.csv files found. Using fidelity_export.csv")
            transformer = FidelityTransformer()
            transformer.transform()
