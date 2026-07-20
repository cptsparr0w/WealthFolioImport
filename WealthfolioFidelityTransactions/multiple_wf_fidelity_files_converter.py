#!/usr/bin/env python3
"""
Combined Fidelity CSV Export for Wealthfolio Import - CORRECTED VERSION
This script combines multiple Fidelity export files into a single file 
compatible with Wealthfolio's CSV import format.
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [FidelityTransformer] - %(message)s'
)
logger = logging.getLogger(__name__)


class FidelityTransformer:
    """High-performance transformer for converting Fidelity CSV exports
       to Wealthfolio-compatible transaction schemas."""

    # === Class Constants ===
    TARGET_COLUMNS = ['Date', 'Type', 'Symbol', 'Quantity', 'Price', 'Amount', 'Currency']
    CLASSIFICATION_MAP: List[Tuple[str, str]] = [
        (r'TAX|FOREIGN TAX', 'TAX'),
        (r'FEE|COMMISSION', 'FEE'),
        (r'DIVIDEND\s*RECEIVED|REINVESTMENT', 'DIVIDEND'),
        (r'REINVESTMENT', 'BUY'),  # Reinvestments are buys
        (r'INTEREST', 'INTEREST'),
        (r'DEPosit|CONTRIBUTION|TRANSFER IN|CREDIT', 'DEPOSIT'),
        (r'WITHDRAWAL|TRANSFER OUT', 'WITHDRAWAL'),
        (r'SPLIT', 'SPLIT'),
        (r'SELL|SOLD', 'SELL'),
        (r'BUY|BOUGHT', 'BUY'),
    ]

    def __init__(self, input_path: str = None, output_path: str = None):
        """Initialize transformer with optional input/output paths."""
        current_dir = Path(os.getcwd())
        if input_path is None:
            csv_files = list(current_dir.glob("History_for_Account_*.csv"))
            if not csv_files:
                self.input_path = current_dir / "fidelity_export.csv"
            else:
                self.input_path = csv_files[0]
            logger.info(f"Found input file: {self.input_path}")
        else:
            self.input_path = Path(input_path)

        if output_path is None:
            stem = self.input_path.stem
            self.output_path = current_dir / f"wealthfolio_import_{stem}.csv"
        else:
            self.output_path = Path(output_path)

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean whitespace and standardize casing."""
        cols_to_upper = ['Action', 'Description', 'Symbol']
        for col in cols_to_upper:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
        return df

    def _handle_options_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggressively normalizes symbols to their base tickers."""
        df['Symbol'] = df['Symbol'].astype(str).str.strip().str.replace(r'^[-\s]+', '', regex=True)
        ticker_extraction_pattern = re.compile(r'^([A-Z\.\d]+)\d{6}[PC]')
        mask = df['Symbol'].str.contains(r'\d{6}[PC]', na=False)
        if mask.any():
            count = mask.sum()
            logger.info(f"Normalizing {count} option symbol strings to base tickers.")
            df.loc[mask, 'Symbol'] = df.loc[mask, 'Symbol'].str.extract(ticker_extraction_pattern, expand=False)
        return df

    def _apply_intelligent_categorization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Uses regex matching to assign Wealthfolio activity types."""
        search_corpus = df['Action'].fillna('') + " " + df['Description'].fillna('')
        conditions, choices = [], []

        for pattern, wealthfolio_type in self.CLASSIFICATION_MAP:
            conditions.append(search_corpus.str.contains(pattern, regex=True, na=False))
            choices.append(wealthfolio_type)

        df['Type'] = np.select(conditions, choices, default='Adjustment')
        return df

    def _handle_adjustments_as_deposits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forces 'Adjustment' (CASH) to 'Deposit' for Wealthfolio compatibility."""
        mask = ((df['Symbol'] == 'CASH') | (df['Symbol'].isna() | (df['Symbol'] == ''))) & (df['Type'] == 'Adjustment')
        if mask.any():
            count = mask.sum()
            logger.info(f"Converting {count} 'Adjustment' (CASH) to 'Deposit'.")
            df.loc[mask, 'Type'] = 'Deposit'
        return df

    def _handle_missing_cost_basis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort by Date to ensure buys are processed before sells."""
        return df.sort_values(by='Date')

    def transform(self) -> None:
        """Main execution pipeline."""
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        try:
            logger.info(f"Starting transformation for {self.input_path}")
            # 1. Read CSV: Skip metadata lines
            df = pd.read_csv(str(self.input_path), skiprows=1, skipfooter=10, engine='python')
            df.columns = df.columns.str.strip()

            # ✅ FIX: Normalize Fidelity column names like "Amount ($)" → "Amount"
            df.columns = df.columns.str.replace(r'\s*\(\$\)', '', regex=True)

            if 'Run Date' in df.columns:
                df = df.rename(columns={'Run Date': 'Date'})
            if df.empty:
                raise ValueError("The input CSV is empty.")
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
            df['Quantity'] = df['Quantity'].abs().fillna(1.0)
            df['Currency'] = 'USD'

            # --- CLEANUP DIVIDEND SPLITS (KEY FIX) ---
            valid_dividend_mask = ~((df['Type'] == 'Dividend') & (df['Price'].isna()) & (df['Quantity'] == 0.0))
            df = df[valid_dividend_mask].copy()
            logger.info(f"Filtered out invalid dividend rows. Remaining: {len(df)}")

            df = self._handle_adjustments_as_deposits(df)
            logger.info("Sorting transactions chronologically.")
            df = self._handle_missing_cost_basis(df)

            # --- FINAL Schema & Cleanup ---
            logger.info("Aligning final schema.")
            df['Symbol'] = df['Symbol'].replace('NAN', 'CASH').fillna('CASH')
            final_df = df[['Date', 'Type', 'Symbol', 'Quantity', 'Price', 'Amount', 'Currency']].copy()

            # === 🚨 CORE FIX: Replace cash symbols with $CASH-USD, preserving real CASH ticker ===
            def is_cash_transaction(row):
                if row['Type'] in ['DEPOSIT', 'WITHDRAWAL', 'FEE', 'TAX', 'INTEREST']:
                    return True
                if (row['Type'] == 'DIVIDEND' and 
                    row['Symbol'] == 'CASH' and 
                    abs(row['Amount']) > 0 and
                    (row['Quantity'] == 0 or pd.isna(row['Quantity']))):
                    return True
                return False

            mask = final_df.apply(is_cash_transaction, axis=1)
            final_df.loc[mask, 'Symbol'] = '$CASH-USD'
            # === ✅ END OF FIX ===

            final_df = final_df.dropna(subset=['Date', 'Type', 'Amount']).copy()
            logger.info(f"After dropping invalid rows: {len(final_df)} entries remaining.")

            output_path = Path(os.getcwd()) / "combined_wealthfolio_import.csv"
            final_df.to_csv(str(output_path), index=False)
            logger.info(f"Success! Combined processed file saved to: {output_path}")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


# === FUNCTION OUTSIDE CLASS (FIXED) ===
def transform_all_fidelity_files():
    """
    Process all Fidelity CSV files in the current directory and combine into one output.
    """
    csv_files = list(Path(os.getcwd()).glob("History_for_Account_*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in current directory.")
        return
    logger.info(f"Found {len(csv_files)} CSV files to combine")

    dataframes = []
    for csv_file in csv_files:
        logger.info(f"Reading and combining {csv_file}")
        try:
            df = pd.read_csv(csv_file, skiprows=1, skipfooter=10, engine='python')
            df.columns = df.columns.str.strip()

            # ✅ FIX: Normalize Fidelity column names like "Amount ($)" → "Amount"
            df.columns = df.columns.str.replace(r'\s*\(\$\)', '', regex=True)

            df['Source_File'] = csv_file.name
            if 'Run Date' in df.columns:
                df = df.rename(columns={'Run Date': 'Date'})
            dataframes.append(df)
        except Exception as e:
            logger.error(f"Failed to process {csv_file}: {e}")

    if not dataframes:
        logger.error("No valid CSV files found")
        return

    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined {len(csv_files)} files into single DataFrame with {len(combined_df)} rows.")

    transformer = FidelityTransformer()
    try:
        combined_df = transformer._preprocess(combined_df)
        combined_df = transformer._handle_options_logic(combined_df)
        combined_df = transformer._apply_intelligent_categorization(combined_df)

        combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce').dt.strftime('%Y-%m-%d 16:00:00')
        combined_df['Amount'] = pd.to_numeric(combined_df['Amount'], errors='coerce').fillna(0.0)
        combined_df['Price'] = combined_df['Price'].fillna(0.0)
        combined_df['Quantity'] = combined_df['Quantity'].abs().fillna(1.0)
        combined_df['Currency'] = 'USD'

        combined_df = transformer._handle_adjustments_as_deposits(combined_df)
        logger.info("Sorting transactions chronologically.")
        combined_df = transformer._handle_missing_cost_basis(combined_df)

        logger.info("Aligning final schema.")
        combined_df['Symbol'] = combined_df['Symbol'].replace('NAN', 'CASH').fillna('CASH')
        final_df = combined_df[['Date', 'Type', 'Symbol', 'Quantity', 'Price', 'Amount', 'Currency']].copy()

        def is_cash_transaction(row):
            if row['Type'] in ['DEPOSIT', 'WITHDRAWAL', 'FEE', 'TAX', 'INTEREST']:
                return True
            if (row['Type'] == 'DIVIDEND' and 
                row['Symbol'] == 'CASH' and 
                abs(row['Amount']) > 0 and
                (row['Quantity'] == 0 or pd.isna(row['Quantity']))):
                return True
            return False

        mask = final_df.apply(is_cash_transaction, axis=1)
        final_df.loc[mask, 'Symbol'] = '$CASH-USD'

        final_df = final_df.dropna(subset=['Date', 'Type', 'Amount']).copy()
        logger.info(f"After dropping invalid rows: {len(final_df)} entries remaining.")

        output_path = Path(os.getcwd()) / "combined_wealthfolio_import.csv"
        final_df.to_csv(str(output_path), index=False)
        logger.info(f"Success! Combined processed file saved to: {output_path}")

    except Exception as e:
        logger.error(f"Combined transformation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        transform_all_fidelity_files()
    else:
        csv_files = list(Path(os.getcwd()).glob("History_for_Account_*.csv"))
        if len(csv_files) > 1:
            logger.info(f"Found {len(csv_files)} CSV files. Processing all...")
            transform_all_fidelity_files()
        elif len(csv_files) == 1:
            logger.info(f"Found single CSV file: {csv_files[0]}")
            transformer = FidelityTransformer(str(csv_files[0]))
            transformer.transform()
        else:
            logger.info("No History_for_Account_*.csv files found.")
