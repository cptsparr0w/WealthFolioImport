# === CORRECTED __main__ BLOCK TO PRODUCE A SINGLE COMBINED FILE ===

if __name__ == "__main__":
    import sys
    
    # This block ensures only ONE combined output file is created.
    # It finds all CSV files, combines them first, then applies the transformation.
    
    logger.info("Starting combined transformation pipeline")
    
    # Find all CSV files in the current directory
    csv_files = glob.glob("History_for_Account_*.csv")
    
    if not csv_files:
        logger.warning("No Fidelity CSV files found. Creating empty output.")
        # Create an empty output file with headers
        combined_df = pd.DataFrame(columns=['Date', 'Type', 'Symbol', 'Quantity', 'Price', 'Amount', 'Description', 'Currency'])
        combined_df.to_csv("combined_wealthfolio_import.csv", index=False)
    else:
        logger.info(f"Found {len(csv_files)} CSV files to combine and process.")
        
        # Create the transformer
        transformer = FidelityTransformer()
        
        # Call a modified transform method that handles multiple files
        try:
            # 1. Manually combine the DataFrames first
            dataframes = []
            for csv_file in csv_files:
                logger.info(f"Reading and combining {csv_file}")
                df = pd.read_csv(csv_file, skiprows=1, skipfooter=10, engine='python')
                df.columns = df.columns.str.strip()
                # Add a source column for traceability
                df['Source_File'] = csv_file
                dataframes.append(df)
            
            # Combine all DataFrames into one
            combined_df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Combined {len(csv_files)} files into single DataFrame with {len(combined_df)} rows.")
            
            # 2. Now apply the transformation to this single combined DataFrame
            # We can reuse some logic from the class methods
            
            # Rename 'Run Date' to 'Date'
            if 'Run Date' in combined_df.columns:
                combined_df = combined_df.rename(columns={'Run Date': 'Date'})
            
            if combined_df.empty:
                raise ValueError("Combined CSV data is empty.")
                
            # Apply preprocessing
            combined_df = transformer._preprocess(combined_df)
            combined_df = transformer._handle_options_logic(combined_df)
            combined_df = transformer._apply_intelligent_categorization(combined_df)
            
            # Date and Amount standardization
            combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce').dt.strftime('%Y-%m-%d 16:00:00')
            combined_df['Amount'] = pd.to_numeric(combined_df['Amount'], errors='coerce').fillna(0.0)
            combined_df['Price'] = combined_df['Price'].fillna(0.0)
            combined_df['Quantity'] = combined_df['Quantity'].fillna(0.0)
            
            # Currency
            logger.info("Setting currency to 'USD' for all transactions.")
            combined_df['Currency'] = 'USD'
            
            # Dividend splits cleanup
            valid_dividend_mask = ~((combined_df['Type'] == 'Dividend') & (combined_df['Price'].isna()) & (combined_df['Quantity'] == 0.0))
            combined_df = combined_df[valid_dividend_mask].copy()
            logger.info(f"Filtered out invalid dividend rows. Remaining: {len(combined_df)}")
            
            # Convert Adjustments to Deposits
            combined_df = transformer._handle_adjustments_as_deposits(combined_df)
            
            # Chronological sort
            logger.info("Sorting transactions chronologically to resolve potential cost-basis mismatches.")
            combined_df = transformer._handle_missing_cost_basis(combined_df)
            
            # Final schema
            logger.info("Aligning final schema.")
            combined_df['Symbol'] = combined_df['Symbol'].replace('NAN', 'CASH').fillna('CASH')
            final_df = combined_df[transformer.TARGET_COLUMNS].dropna(subset=['Date', 'Type', 'Amount']).copy()
            logger.info(f"After dropping invalid rows: {len(final_df)} entries remaining.")
            
            # Validate and Export (Single File)
            transformer._validate(final_df)
            output_path = Path(os.getcwd()) / "combined_wealthfolio_import.csv"
            final_df.to_csv(str(output_path), index=False)
            logger.info(f"Success! Combined processed file saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
