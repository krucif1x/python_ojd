# generate_perfectly_balanced_dataset.py
import pandas as pd
import numpy as np
import random

def generate_perfect_dataset(samples_per_class=25, output_file='manual_labels.csv'):
    """
    Generates a perfectly balanced dataset.
    
    Args:
        samples_per_class: How many plates to generate for EACH (Region + Vehicle) combo.
                           Default 25 -> 100 plates per Region -> ~3700 total.
    """
    print(f"🔧 Generating PERFECTLY BALANCED dataset...")
    print(f"   Target: {samples_per_class} samples per Vehicle Type per Region.")
    
    data = []
    
    # --- DEFINITIONS ---
    # 1. Non-Jakarta Provinces (Prefix -> Region Name)
    # Note: These prefixes determine the region directly.
    provinces = {
        'A': 'Banten', 'D': 'Bandung Raya', 'E': 'Cirebon/Indramayu', 
        'F': 'Bogor/Sukabumi', 'G': 'Pekalongan', 'H': 'Semarang', 
        'K': 'Pati', 'L': 'Surabaya', 'M': 'Madura', 'N': 'Malang', 
        'P': 'Besuki', 'R': 'Banyumas', 'S': 'Bojonegoro', 
        'T': 'Karawang/Purwakarta', 'W': 'Sidoarjo', 'Z': 'Tasikmalaya', 
        'AA': 'Kedu', 'AB': 'Yogyakarta', 'AD': 'Surakarta', 
        'AE': 'Madiun', 'AG': 'Kediri', 'DK': 'Bali', 
        'DA': 'Kalsel', 'KB': 'Kalbar', 'DB': 'Sulut', 'DD': 'Sulsel'
    }
    
    # 2. Jakarta Sub-Regions (Suffix First Letter -> Region Name)
    # Prefix is always 'B'
    jakarta_suffixes = {
        'B': 'Jakarta Barat', 'P': 'Jakarta Pusat', 'S': 'Jakarta Selatan',
        'T': 'Jakarta Timur', 'U': 'Jakarta Utara', 'E': 'Depok', 
        'Z': 'Depok', 'K': 'Bekasi Kota', 'F': 'Bekasi Kab', 
        'C': 'Tangerang Kota', 'W': 'Tangsel'
    }
    
    vehicle_types = ['Private Car', 'Motorcycle', 'Bus', 'Truck/Commercial']

    # --- GENERATOR FUNCTION ---
    def make_plate(prefix, region_name, v_type, forced_suffix_char=None):
        # 1. Generate Number based on Vehicle Type
        if v_type == 'Private Car': num = random.randint(1, 1999)
        elif v_type == 'Motorcycle': num = random.randint(2000, 6999)
        elif v_type == 'Bus': num = random.randint(7000, 7999)
        else: num = random.randint(8000, 9999)
        
        # 2. Generate Suffix
        # If Jakarta, first letter is forced. Else, random.
        if forced_suffix_char:
            # Suffix = Forced Char + 1 or 2 Random Chars
            suffix = forced_suffix_char + "".join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=random.randint(1, 2)))
        else:
            suffix = "".join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=random.randint(2, 3)))
            
        return {
            'plate': f"{prefix}{num}{suffix}",
            'prefix': prefix,
            'number': num,
            'suffix': suffix,
            'province': 'Jadetabek' if prefix == 'B' else region_name, # Simplified Province Logic
            'region': region_name,
            'vehicle_type': v_type
        }

    # --- MAIN LOOPS ---

    # Loop 1: Jakarta Regions (The "B" Plates)
    print("   ↳ Generating Jakarta (B) Sub-regions...")
    for suffix_char, region_name in jakarta_suffixes.items():
        for v_type in vehicle_types:
            for _ in range(samples_per_class):
                data.append(make_plate('B', region_name, v_type, forced_suffix_char=suffix_char))

    # Loop 2: Other Provinces
    print("   ↳ Generating Provincial Regions...")
    for prefix, region_name in provinces.items():
        for v_type in vehicle_types:
            for _ in range(samples_per_class):
                data.append(make_plate(prefix, region_name, v_type))

    # --- SAVE ---
    df = pd.DataFrame(data)
    # Shuffle the data so the neural network doesn't learn order patterns
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Verify Counts
    print(f"\n✅ Generated {len(df)} Total Plates.")
    print("-" * 40)
    print(f"   Total 'B' (Jakarta) Plates: {len(df[df['prefix']=='B'])}")
    print(f"   Total Non-Jakarta Plates:   {len(df[df['prefix']!='B'])}")
    print("-" * 40)
    print("   Breakdown per Vehicle Type:")
    print(df['vehicle_type'].value_counts())
    
    df.to_csv(output_file, index=False)
    print(f"\n💾 Saved to {output_file}")

if __name__ == "__main__":
    # 25 samples * 4 vehicles * 37 regions = 3700 total
    generate_perfect_dataset(samples_per_class=25, output_file='/home/pcsistem/Documents/carlo_ojd_ml/plate_text_dataset/manual_labels.csv')