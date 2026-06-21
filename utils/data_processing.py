import os
import pandas as pd
import re
import pickle
from PyPDF2 import PdfReader
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import gzip

def extract_data_from_pdfs(pdf_files):
    """Extract cutoff data from PDF files containing polytechnic cutoffs."""
    all_data = []
    
    for pdf_file in pdf_files:
        print(f"Processing {os.path.basename(pdf_file)}...")
        
        # Extract year from filename
        year_match = re.search(r'(\d{4})', os.path.basename(pdf_file))
        current_year = year_match.group(1) if year_match else "2023"
        
        # Determine CAP round from filename
        round_match = re.search(r'(CAP|cap)\s*(\d)', os.path.basename(pdf_file), re.IGNORECASE)
        cap_round = round_match.group(2) if round_match else "1"
        
        # Read PDF
        try:
            # Check if file exists
            if not os.path.exists(pdf_file):
                print(f"Warning: File {pdf_file} does not exist")
                continue
                
            reader = PdfReader(pdf_file)
            print(f"PDF has {len(reader.pages)} pages")
            
            current_college = None
            college_type = None
            location = None
            
            for page_num, page in enumerate(reader.pages):
                print(f"  Processing page {page_num+1}/{len(reader.pages)}")
                try:
                    text = page.extract_text()
                    lines = text.split('\n')
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Skip header lines
                        if any(x in line for x in ['GOVERNMENT OF MAHARASHTRA', 'State Common Entrance Test Cell', 'Provisional Cutoff']):
                            continue
                        
                        # Extract college details
                        college_match = re.search(r'(\d{4})\s+(.*?)\s+\((.*?)\)', line)
                        if college_match:
                            college_id = college_match.group(1)
                            current_college = college_match.group(2).strip()
                            location = college_match.group(3).strip()
                            
                            # Determine college type
                            if "Govt" in current_college or "Government" in current_college:
                                college_type = "Government"
                            elif "Private" in current_college:
                                college_type = "Private"
                            elif "Aided" in current_college:
                                college_type = "Aided"
                            else:
                                college_type = "Private"  # Default
                            
                            print(f"    Found college: {current_college}")
                            continue
                        
                        # Extract branch and cutoff data
                        if current_college:
                            # Try multiple regex patterns to match cutoff data
                            patterns = [
                                r'(\d{4})\s+(.*?)\s+(\w+)\s+(\d+\.\d+)',  # Standard pattern
                                r'(\d{4})\s+(.*?)\s+(\w+)\s+(\d+)',        # Integer cutoff
                                r'(\d+)\s+(.*?)\s+(\w+)\s+(\d+\.\d+)'      # Different ID format
                            ]
                            
                            for pattern in patterns:
                                cutoff_match = re.search(pattern, line)
                                if cutoff_match:
                                    branch_code = cutoff_match.group(1)
                                    branch_name = cutoff_match.group(2).strip()
                                    category = cutoff_match.group(3)
                                    
                                    try:
                                        cutoff = float(cutoff_match.group(4))
                                        
                                        # Add to data
                                        all_data.append({
                                            'college_name': current_college,
                                            'branch': branch_name,
                                            'category': category,
                                            'cutoff': cutoff,
                                            'location': location,
                                            'college_type': college_type,
                                            'year': current_year,
                                            'round': cap_round
                                        })
                                    except ValueError:
                                        print(f"    Warning: Could not convert cutoff to float: {cutoff_match.group(4)}")
                                    
                                    break  # Exit pattern loop if matched
                except Exception as e:
                    print(f"  Error processing page {page_num+1}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            continue
    
    # Convert to DataFrame
    print(f"Extracted {len(all_data)} records from all PDFs")
    df = pd.DataFrame(all_data)
    return df

def clean_cutoff_data(df):
    """Clean and process the extracted cutoff data."""
    if df.empty:
        print("Error: No data to clean")
        return df
    
    print(f"Cleaning {len(df)} records...")
    
    # Remove duplicates
    df = df.drop_duplicates()
    print(f"After removing duplicates: {len(df)} records")
    
    # Clean branch names
    df['branch'] = df['branch'].str.strip()
    
    # Clean categories
    df['category'] = df['category'].str.upper()
    
    # Convert cutoff to float
    df['cutoff'] = pd.to_numeric(df['cutoff'], errors='coerce')
    
    # Drop rows with missing cutoffs
    df = df.dropna(subset=['cutoff'])
    print(f"After dropping rows with missing cutoffs: {len(df)} records")
    
    # Extract city from location
    df['city'] = df['location'].apply(lambda x: x.split(',')[-1].strip() if isinstance(x, str) and ',' in x else x)
    
    # Clean college type
    df['college_type'] = df['college_type'].fillna('Private')
    
    return df

def train_model(cutoff_data):
    """Train a Random Forest model to predict cutoffs."""
    df = cutoff_data.copy()
    
    print(f"Training model with {len(df)} records...")
    
    # Prepare features and target
    features = df[['college_name', 'branch', 'category', 'college_type', 'city', 'year', 'round']]
    target = df['cutoff']
    
    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(features)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        encoded_features, target, test_size=0.2, random_state=42
    )
    
    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Model Training Score: {train_score:.4f}")
    print(f"Model Testing Score: {test_score:.4f}")
    
    return model, encoder

def process_pdf_data(pdf_files):
    """Process PDF files, extract data, clean it, and train a model."""
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Extract data from PDFs
    print("Extracting data from PDFs...")
    cutoff_data = extract_data_from_pdfs(pdf_files)
    
    if cutoff_data.empty:
        print("Error: No data extracted from PDFs")
        return {
            'records_processed': 0,
            'files_processed': len(pdf_files),
            'model_score': 0
        }
    
    # Save raw extracted data
    raw_data_path = os.path.join('data', 'raw_cutoff_data_10th.csv')
    cutoff_data.to_csv(raw_data_path, index=False)
    print(f"Raw data saved to {raw_data_path}")
    
    # Clean the data
    print("Cleaning and processing data...")
    cleaned_data = clean_cutoff_data(cutoff_data)
    
    if cleaned_data.empty:
        print("Error: No data after cleaning")
        return {
            'records_processed': 0,
            'files_processed': len(pdf_files),
            'model_score': 0
        }
    
    # Save cleaned data
    processed_data_path = os.path.join('data', 'processed_cutoff_data.csv')
    cleaned_data.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}")
    
    # Train model
    print("Training prediction model...")
    model, encoder = train_model(cleaned_data)
    
    # Save model and encoder
    model_path = os.path.join('models', 'polytech_predictor_model.pkl.gz')
    encoder_path = os.path.join('models', 'feature_encoder.pkl')
    
    with gzip.open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with gzip.open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)
    
    print(f"Model saved to {model_path}")
    print(f"Encoder saved to {encoder_path}")
    
    model_score = model.score(encoder.transform(cleaned_data[['college_name', 'branch', 'category', 'college_type', 'city', 'year', 'round']]), cleaned_data['cutoff'])
    
    return {
        'records_processed': len(cleaned_data),
        'files_processed': len(pdf_files),
        'model_score': model_score
    } 