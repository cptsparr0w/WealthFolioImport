#!/usr/bin/env python3
"""
Transform Charles Schwab transaction CSVs into Wealthfol-compatible format.

Designed for:
- HSA_Brokerage_*.csv (HSA account)
- Rollover_IRA_*.csv (IRA/401k)

Output: combined_wealthfolio_import.csv
"""

import os
import glob
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# === HELPER FUNCTIONS =========================================================

def _clean_currency(val):
    """Convert '$1,234.56' / '-$100.00' to float."""
    if pd.isna(val) or not val:
        return 0.0
    try:
        return float(str(val).replace('$', '').replace(',', ''))
    except (ValueError, AttributeError):
        return 0.0

def _clean_date(val):
    """Normalize dates like '09/16/2025 as of 09/15/2025' → 'YYYY-MM-DD'"""
    if pd.isna(val):
        return ''
    d = str(val).split(' as of')[0].strip()
    try:
        return pd.to_datetime(d, format='%m/%d/%Y').strftime('%Y-%m-%d 16:00:00')
    except Exception:
        return ''

def _resolve_type(action, description, symbol):
    """Map Schwab Actions to standardized types (Wealthfol-friendly)."""
    action = str(action or '').strip().lower()
    description = str(description or '').lower()
    symbol = str(symbol or '').upper()

    # Cash-level entries
    if 'bank int' in description:
        return 'Interest'
    if any(x in action for x in ['mmda', 'cash alternatives']):
        return 'Deposit' if '>0' not in action else 'Withdrawal'

    # Drills for dividend types
    if any(x in action for x in ['qualified div', 'special qual div']):
        return 'Dividend'
    if 'non-qualified div' in action:
        return 'NonDividendDistribution'
    if 'adr mgmt fee' in action or 'adr fee' in description:
        return 'Fee'
    if 'foreign tax' in action.lower() or 'fx adj' in description:
        return 'ForeignTaxPaid'

    # Reinvestment
    if any(x in action for x in ['reinvest', 'qual div reinvest']):
        return 'DividendReinvest'

    # Stock moves
    if action == 'buy':
        return 'Buy'
    if action == 'sell':
        return 'Sell'

    # Cash in lieu, reverse splits
    if any(x in action.lower() for x in ['cash in lie', 'reverse split']):
        return 'CashInLieu'

    # Internal transfer / transfer in/out
    if any(x in action for x in ['journaled shares', 'internal transfer']):
        return 'Transfer'

    # Catch-all (e.g., options, mergers)
    if any(x in description for x in ['merger', 'dividend reinvest']):
        return 'DividendReinvest'

    # Unknown action → classify as 'Deposit' for safe import
    if not symbol or symbol == '':
        return 'Deposit'

    logger.warning(f"Unknown action/type mapping for: {action}, desc='{description}', symbol={symbol}")
    return 'Other'


def _is_cash_action(action, symbol):
    """Detect if transaction affects cash balance only (e.g., sweep, interest)."""
    action = str(action or '').lower()
    symbol = str(symbol or '').upper()

    # Schwab’s sweep symbols
    if any(x in symbol for x in ['MMDA', 'CASH']):
        return True
    # Sweep/interest texts
    if any(x in action for x in ['bank int', 'cash alternatives']):
        return True
    # Empty symbol + cash description
    if not symbol and 'transfer' in action:
        return True

    return False


class SchwabTransformer:
    """Transforms Schwab CSV data to Wealthfol schema."""

    TARGET_COLUMNS = [
        'Date',
        'Type',
        'Symbol',
        'Quantity',
        'Price',
        'Amount',
        'Description',
        'Currency'
    ]

    def __init__(self):
        self.mapped_actions = set()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main transformation pipeline."""
        # 1. Clean and prep columns
        df = df.copy()
        df.columns = [col.strip().replace('"', '') for col in df.columns]
        logger.info(f"Columns: {df.columns.tolist()}")

        # 2. Normalize fields
        df['Date'] = df['Date'].apply(_clean_date)
        df['Amount'] = df['Amount'].apply(_clean_currency)
        df['Price'] = df['Price'].apply(_clean_currency).fillna(0.0)
        # Handle Quantity as number or empty
        df['Quantity'] = pd.to_numeric(
            df['Quantity'].astype(str).str.replace(',', ''), 
            errors='coerce'
        ).fillna(0.0)
        df['Description'] = df['Description'].astype(str)

        # 3. Type Mapping (critical!)
        df['Type'] = df.apply(
            lambda row: _resolve_type(row['Action'], row['Description'], row['Symbol']),
            axis=1
        )

        # 4. Symbol Normalization (empty → CASH)
        df['Symbol'] = df['Symbol'].astype(str).replace('', 'CASH').replace('NAN', 'CASH')
        df.loc[df['Symbol'].str.contains(r'MMDA|BANK INT', case=False, regex=True), 'Symbol'] = 'CASH'

        # 5. Special handling: Schwab combine reinvestments in single row → separate into two (Dividend + Buy)
        # but for simplicity, keep as DividendReinvest and let Wealthfol re-split if needed
        # (optional post-processing step)
        
        # 6. Cash-like actions: Quantity=0 & Price=0 → Amount reflects net impact
        for idx, row in df.iterrows():
            if _is_cash_action(row['Action'], row['Symbol']):
                # Skip price/quantity cleaning for cash entries
                continue
            # If not cash, ensure consistency with transaction type:
            if row['Type'] == 'Dividend' and row['Quantity'] != 0.0:
                logger.warning(f"Unexpected non-zero quantity for dividend: {row}")
                df.at[idx, 'Quantity'] = 0.0
            if row['Type'] == 'Buy' and row['Quantity'] <= 0:
                logger.warning(f"Zero/negative quantity for buy: {row}")
            if row['Type'] == 'Sell' and row['Quantity'] >= 0:
                # Schwab may use negative quantity in "Shares Sold"
                df.at[idx, 'Quantity'] = abs(row['Quantity'])
            if row['Type'] == 'Sell':
                # Amount should be positive when symbol is sold; double-check sign
                if row['Amount'] > 0:
                    df.at[idx, 'Amount'] = abs(row['Amount'])
                else:
                    df.at[idx, 'Amount'] = -abs(row['Amount'])

        # 7. Final clean-up
        df['Currency'] = 'USD'
        df['Amount'] = df['Amount'].fillna(0.0)
        # Remove zero-amount entries unless cash/interest (keep for cost-basis completeness)
        df = df[~((df['Amount'] == 0.0) & (df['Type'] != 'Deposit') & (df['Type'] != 'Interest'))].copy()

        # 8. Chronological sort
        df = df.sort_values(['Date']).reset_index(drop=True)

        # 9. Select final columns
        return df[[
            'Date', 'Type', 'Symbol', 'Quantity',
            'Price', 'Amount', 'Description', 'Currency'
        ]].dropna(subset=['Date', 'Type']).copy()


if __name__ == "__main__":
    # === CORRECTED MAIN BLOCK FOR SCHWAB EXTRACTS ===
    
    logger.info("Starting Schwab transformation pipeline...")

    # Find both Schwab-style CSVs
    csv_files = sorted(glob.glob("HSA_Brokerage_*.csv") + glob.glob("Rollover_IRA_*.csv"))
    if not csv_files:
        logger.warning("No Schwab CSVs found. Creating empty output.")
        pd.DataFrame(columns=SchwabTransformer.TARGET_COLUMNS).to_csv(
            "combined_wealthfolio_import.csv", index=False
        )
        exit(0)

    logger.info(f"Found {len(csv_files)} Schwab files: {csv_files}")

    transformer = SchwabTransformer()
    dataframes = []

    for csv_file in csv_files:
        logger.info(f"Processing {csv_file}...")
        df = pd.read_csv(csv_file)
        # Drop the leading/trailing quotes if any
        df.columns = [col.strip().strip('"') for col in df.columns]
        # Handle embedded quotes safely
        df = df.replace(r'"', '', regex=True)
        
        processed_df = transformer.transform(df)
        dataframes.append(processed_df)

    # Combine & finalize
    final_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined {len(csv_files)} files → {len(final_df)} rows")

    # Optional: deduplicate (if same event appears in both HSA/IRA)
    if not final_df.duplicated(subset=['Date', 'Type', 'Symbol', 'Quantity', 'Amount']).all():
        logger.warning("Detected possible duplicates across accounts.")
    
    # Drop any rows missing required fields
    final_df = final_df.dropna(subset=['Date', 'Type', 'Amount'])
    
    output_path = Path(os.getcwd()) / "combined_wealthfolio_import.csv"
    final_df.to_csv(str(output_path), index=False)
    logger.info(f"✅ Success! Output saved to: {output_path}")
