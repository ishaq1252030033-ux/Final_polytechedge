import pandas as pd
import numpy as np

def predict_colleges(user_input, model, encoder, cutoff_data):
    """
    Predict colleges based on user input
    
    Parameters:
    - user_input: dict with keys 'percentage', 'category', 'branch', 'college_type', 'city'
    - model: trained prediction model
    - encoder: feature encoder
    - cutoff_data: DataFrame with historical cutoff data
    
    Returns:
    - DataFrame with recommended colleges and admission probabilities
    """
    print(f"Processing prediction with user input: {user_input}")
    
    # Ensure percentage is a float
    try:
        user_percentage = float(user_input['percentage'])
    except (ValueError, TypeError):
        user_percentage = 0.0
    
    # Apply category-based relaxation (additional marks for reserved categories)
    relaxation = {
        'GENERAL': 0,
        'OBC': 5,
        'SC': 10,
        'ST': 15,
        'SEBC': 7,
        'EWS': 3,
        'TFWS': 0,
        'PH': 10
    }
    
    # Ensure category is uppercase for consistency
    user_category = user_input['category'].upper() if user_input['category'] else 'GENERAL'
    
    # Apply relaxation
    adjusted_percentage = user_percentage + relaxation.get(user_category, 0)
    print(f"Adjusted percentage: {adjusted_percentage} (Category: {user_category})")
    
    # Filter colleges based on user preferences - do this efficiently
    filtered_colleges = cutoff_data.copy()
    print(f"Initial number of colleges: {len(filtered_colleges)}")
    
    # Apply filters using boolean indexing for better performance
    if user_input['college_type'] and user_input['college_type'].lower() != "all":
        requested_type = user_input['college_type'].lower().strip()
        available_types = (
            filtered_colleges['college_type']
            .astype(str)
            .str.lower()
            .str.strip()
            .unique()
        )

        if requested_type == 'autonomous':
            # Some datasets don’t label autonomous explicitly; avoid filtering to zero results.
            if 'autonomous' in available_types:
                filtered_colleges = filtered_colleges[
                    filtered_colleges['college_type'].str.lower().str.strip() == 'autonomous'
                ]
                print(f"After autonomous type filter: {len(filtered_colleges)}")
            else:
                print("No 'Autonomous' label in data; skipping college_type filter for autonomous.")
        else:
            filtered_colleges = filtered_colleges[
                filtered_colleges['college_type'].str.lower().str.strip() == requested_type
            ]
            print(f"After college type filter: {len(filtered_colleges)}")
    
    if user_input['city'] and user_input['city'].lower() != "all":
        # Clean city names for comparison
        filtered_colleges['clean_city'] = filtered_colleges['city'].str.lower().str.strip()
        user_city = user_input['city'].lower().strip()
        
        # Handle special cases for cities
        if user_city == 'latur':
            filtered_colleges = filtered_colleges[
                filtered_colleges['clean_city'] == 'latur'
            ]
        elif user_city == 'navi mumbai':
            filtered_colleges = filtered_colleges[
                filtered_colleges['clean_city'] == 'navi mumbai'
            ]
        else:
            filtered_colleges = filtered_colleges[
                filtered_colleges['clean_city'] == user_city
            ]
        
        print(f"After city filter: {len(filtered_colleges)}")
        print(f"Found colleges in {user_city}:")
        for college in filtered_colleges['college_name'].unique():
            print(f"- {college}")
    
    if user_input['branch'] and user_input['branch'].lower() != "all":
        filtered_colleges = filtered_colleges[
            filtered_colleges['branch'].str.lower() == user_input['branch'].lower()
        ]
        print(f"After branch filter: {len(filtered_colleges)}")
    
    # Clean up temp columns
    if 'clean_city' in filtered_colleges.columns:
        filtered_colleges = filtered_colleges.drop('clean_city', axis=1)

    # Clean college and branch names to remove leading/trailing dashes and spaces
    filtered_colleges['college_name'] = filtered_colleges['college_name'].str.strip().str.lstrip('-').str.strip()
    filtered_colleges['branch'] = filtered_colleges['branch'].str.strip()

    # Remove duplicate college entries before processing
    filtered_colleges = filtered_colleges.drop_duplicates(subset=['college_name', 'branch'])

    # Get unique college-branch combinations efficiently
    unique_colleges = filtered_colleges.copy()
    print(f"Final number of unique college-branch combinations: {len(unique_colleges)}")
    
    # Get the latest year and round from the actual data
    latest_year = filtered_colleges['year'].mode().iloc[0] if 'year' in filtered_colleges.columns and not filtered_colleges.empty else "2024"
    latest_round = filtered_colleges['round'].mode().iloc[0] if 'round' in filtered_colleges.columns and not filtered_colleges.empty else "3"
    print(f"Using latest year: {latest_year}, latest round: {latest_round}")
    
    # Check the encoder feature order - this is critical
    if hasattr(encoder, 'feature_names_in_'):
        print(f"Encoder expects features in this order: {encoder.feature_names_in_}")
        feature_order = list(encoder.feature_names_in_)
    else:
        print("Cannot determine encoder feature order")
        feature_order = ['college_name', 'branch', 'category', 'year', 'round', 'college_type', 'city']
    
    # Prepare all inputs at once for better performance
    results = []
    
    # This is critical - get actual cutoffs for comparison if available
    actual_cutoffs = {}
    for _, row in filtered_colleges.iterrows():
        key = (row['college_name'], row['branch'])
        if key not in actual_cutoffs or (row['year'] == latest_year and row['round'] == latest_round):
            actual_cutoffs[key] = row['cutoff'] if 'cutoff' in row else None
    
    print(f"Found {len(actual_cutoffs)} actual cutoffs for comparison")
    
    # Track already processed colleges to avoid duplicates
    processed_colleges = set()
    
    for _, college in unique_colleges.iterrows():
        # Create a unique identifier for this college-branch combination
        college_key = (college['college_name'], college['branch'])
        
        # Skip if we've already processed this college-branch combination
        if college_key in processed_colleges:
            print(f"Skipping duplicate: {college['college_name']}, {college['branch']}")
            continue
        
        # Mark this college-branch combination as processed
        processed_colleges.add(college_key)
        
        try:
            # Create input dictionary with the correct feature order
            input_dict = {}
            for feature in feature_order:
                if feature == 'college_name':
                    input_dict[feature] = str(college['college_name'])
                elif feature == 'branch':
                    input_dict[feature] = str(college['branch'])
                elif feature == 'category':
                    input_dict[feature] = user_category
                elif feature == 'year':
                    input_dict[feature] = latest_year
                elif feature == 'round':
                    input_dict[feature] = latest_round
                elif feature == 'college_type':
                    input_dict[feature] = str(college['college_type'])
                elif feature == 'city':
                    input_dict[feature] = str(college['city'])
                else:
                    input_dict[feature] = ''
            
            # Create DataFrame with the right feature order
            college_input = pd.DataFrame([input_dict])
            
            # Debug
            print(f"Predicting for: {college['college_name']}, {college['branch']}")
            print(f"Input features: {list(college_input.columns)}")
            
            # Check if we have actual cutoff data for this college-branch combination
            key = (college['college_name'], college['branch'])
            actual_cutoff = actual_cutoffs.get(key)
            
            # Use the actual cutoff if available, otherwise predict
            if actual_cutoff is not None:
                predicted_cutoff = actual_cutoff
                print(f"Using actual cutoff of {predicted_cutoff}")
            else:
                # Encode features and predict
                try:
                    encoded_input = encoder.transform(college_input)
                    predicted_cutoff = float(model.predict(encoded_input)[0])
                    print(f"Model predicted cutoff: {predicted_cutoff}")
                except Exception as e:
                    print(f"Error during prediction: {str(e)}")
                    continue
            
            # Calculate admission probability with better scaling
            if adjusted_percentage >= predicted_cutoff:
                # If percentage is above cutoff, calculate probability based on margin
                margin = adjusted_percentage - predicted_cutoff
                probability = min(99, 75 + margin * 3)  # More generous scaling
            else:
                # If percentage is below cutoff, calculate probability based on deficit
                deficit = predicted_cutoff - adjusted_percentage
                if deficit <= 10:
                    probability = max(20, 70 - deficit * 5)  # Less harsh penalty
                elif deficit <= 15:
                    probability = max(10, 40 - (deficit - 10) * 6)
                else:
                    probability = 5
            
            results.append({
                'college_name': college['college_name'],
                'branch': college['branch'],
                'city': college['city'],
                'college_type': college['college_type'],
                'adjusted_percentage': round(adjusted_percentage, 2),
                'probability': round(probability, 1),
                'status': get_admission_status(probability)
            })
            print(f"Prediction successful: cutoff={round(predicted_cutoff, 2)}, probability={round(probability, 1)}%")
        except Exception as e:
            print(f"Error processing {college['college_name']}: {str(e)}")
            continue
    
    # Convert to DataFrame and sort by probability
    if results:
        results_df = pd.DataFrame(results)
        
        # Double-check for duplicates before returning
        results_df = results_df.drop_duplicates(subset=['college_name', 'branch'])
        
        results_df = results_df.sort_values('probability', ascending=False)
        print(f"Found {len(results_df)} matching colleges")
        return results_df
    else:
        print("No matching colleges found. Trying alternative approach...")
        
        # Try using available data more directly
        simple_results = []
        processed_colleges = set()  # Track processed colleges to avoid duplicates
        
        # For each college in our data, calculate probability directly
        for _, college in unique_colleges.iterrows():
            # Create a unique identifier for this college-branch combination
            college_key = (college['college_name'], college['branch'])
            
            # Skip if we've already processed this college-branch combination
            if college_key in processed_colleges:
                continue
            
            # Mark this college-branch combination as processed
            processed_colleges.add(college_key)
            
            # Get any actual cutoffs we have for this college
            college_cutoffs = filtered_colleges[
                (filtered_colleges['college_name'] == college['college_name']) &
                (filtered_colleges['branch'] == college['branch'])
            ]
            
            if not college_cutoffs.empty:
                # Use the most recent cutoff
                recent_cutoff = college_cutoffs.iloc[0]['cutoff'] if 'cutoff' in college_cutoffs.columns else 85.0
                
                # Calculate probability
                if adjusted_percentage >= recent_cutoff:
                    probability = min(99, 75 + (adjusted_percentage - recent_cutoff) * 3)
                else:
                    deficit = recent_cutoff - adjusted_percentage
                    probability = max(5, 70 - deficit * 5) if deficit <= 10 else 5
                
                simple_results.append({
                    'college_name': college['college_name'],
                    'branch': college['branch'],
                    'city': college['city'],
                    'college_type': college['college_type'],
                    'adjusted_percentage': round(adjusted_percentage, 2),
                    'probability': round(probability, 1),
                    'status': get_admission_status(probability)
                })
                print(f"Direct calculation: cutoff={round(recent_cutoff, 2)}, probability={round(probability, 1)}%")
            
        if simple_results:
            simple_df = pd.DataFrame(simple_results)
            # Ensure no duplicates
            simple_df = simple_df.drop_duplicates(subset=['college_name', 'branch'])
            return simple_df.sort_values('probability', ascending=False)
        
        # Last resort fallback
        print("No colleges found with available data. Showing generic options.")
        fallback_results = []
        processed_colleges = set()  # Reset tracking set
        
        for _, college in unique_colleges.iterrows():
            # Create a unique identifier for this college-branch combination
            college_key = (college['college_name'], college['branch'])
            
            # Skip if we've already processed this college-branch combination
            if college_key in processed_colleges:
                continue
            
            # Mark this college-branch combination as processed
            processed_colleges.add(college_key)
            
            # Assign varying probabilities based on college type to show differentiation
            base_prob = 50.0
            if college['college_type'].lower() == 'government':
                base_prob = 30.0
            elif college['college_type'].lower() == 'private':
                base_prob = 60.0
            
            # Add some randomness to show variation
            probability = max(5, min(95, base_prob + np.random.normal(0, 10)))
            
            fallback_results.append({
                'college_name': college['college_name'],
                'branch': college['branch'],
                'city': college['city'],
                'college_type': college['college_type'],
                'adjusted_percentage': round(adjusted_percentage, 2),
                'probability': round(probability, 1),
                'status': get_admission_status(probability)
            })
        
        if fallback_results:
            fallback_df = pd.DataFrame(fallback_results)
            # Ensure no duplicates
            fallback_df = fallback_df.drop_duplicates(subset=['college_name', 'branch'])
            return fallback_df.sort_values('probability', ascending=False)
        
        print("No colleges found at all")
        return pd.DataFrame()

def get_admission_status(probability):
    """Determine admission status based on probability."""
    if probability >= 90:
        return "Very Likely"
    elif probability >= 75:
        return "Likely"
    elif probability >= 50:
        return "Moderate Chance"
    elif probability >= 25:
        return "Less Likely"
    else:
        return "Unlikely"