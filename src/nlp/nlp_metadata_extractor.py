# indonesian_plate_nlp_manual_labels.py
"""
Indonesian License Plate NLP Metadata Extractor
USES ONLY MANUAL LABELS - No rule-based, no synthetic generation during training
"""

import re
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class ManualLabelsPlateExtractor:
    def __init__(self, manual_labels_path, model_path=None):
        """
        Initialize with manual labels CSV only
        
        Args:
            manual_labels_path: Path to manual_labels.csv
            model_path: Path to pre-trained model (optional)
        """
        self.manual_labels_path = manual_labels_path
        self.model_path = model_path
        
        # Initialize knowledge base for prediction fallback
        self.initialize_knowledge_base()
        
        # Encoders
        self.province_encoder = LabelEncoder()
        self.region_encoder = LabelEncoder()
        self.vehicle_encoder = LabelEncoder()
        
        # Initialize models
        self.initialize_models()
        
        # Load training data from manual labels ONLY
        self.training_data = None
        if manual_labels_path:
            self.training_data = self.load_manual_labels()

    def initialize_knowledge_base(self):
        """Knowledge base for fallback predictions only (not for training)"""
        self.province_map = {
            'A': 'Banten', 'B': 'Jadetabek', 'D': 'Bandung Raya', 
            'E': 'Cirebon/Indramayu', 'F': 'Bogor/Sukabumi', 
            'G': 'Pekalongan', 'H': 'Semarang', 'K': 'Pati',
            'L': 'Surabaya', 'M': 'Madura', 'N': 'Malang', 
            'P': 'Besuki', 'R': 'Banyumas', 'S': 'Bojonegoro', 
            'T': 'Karawang/Purwakarta', 'W': 'Sidoarjo', 
            'Z': 'Tasikmalaya', 'AA': 'Kedu', 'AB': 'Yogyakarta', 
            'AD': 'Surakarta', 'AE': 'Madiun', 'AG': 'Kediri',
            'DK': 'Bali', 'DA': 'Kalsel', 'KB': 'Kalbar', 
            'DB': 'Sulut', 'DD': 'Sulsel'
        }
        self.jakarta_suffix_map = {
            'B': 'Jakarta Barat', 'P': 'Jakarta Pusat', 'S': 'Jakarta Selatan',
            'T': 'Jakarta Timur', 'U': 'Jakarta Utara', 'E': 'Depok', 
            'Z': 'Depok', 'K': 'Bekasi Kota', 'F': 'Bekasi Kab', 
            'C': 'Tangerang Kota', 'W': 'Tangsel'
        }

    def initialize_models(self):
        """Initialize ML models with regularization to prevent overfitting"""
        
        # Province classifier
        self.province_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
        
        # Region classifier
        self.region_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
        
        # Vehicle type classifier
        self.vehicle_classifier = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=3,
            subsample=0.8,  # Use 80% of samples per tree
            random_state=42
        )
        
        # TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 4),
            max_features=600,
            analyzer='char',
            lowercase=False
        )

    def load_manual_labels(self):
        """
        Load manual labels CSV - the ONLY source of training data
        NO rule-based generation, NO synthetic data here
        """
        if not os.path.exists(self.manual_labels_path):
            raise FileNotFoundError(f"Manual labels not found: {self.manual_labels_path}")
        
        print(f"📂 Loading manual labels from: {self.manual_labels_path}")
        
        df = pd.read_csv(self.manual_labels_path)
        
        # Validate required columns
        required_cols = ['plate', 'province', 'region', 'vehicle_type']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Extract data
        texts = df['plate'].astype(str).tolist()
        provinces = df['province'].astype(str).tolist()
        regions = df['region'].astype(str).tolist()
        vehicle_types = df['vehicle_type'].astype(str).tolist()
        
        print(f"✅ Loaded {len(texts)} manually labeled plates")
        print(f"\n📊 Dataset Composition:")
        print(f"   Unique provinces: {len(set(provinces))}")
        print(f"   Unique regions: {len(set(regions))}")
        print(f"   Unique vehicle types: {len(set(vehicle_types))}")
        
        print(f"\n   Vehicle Type Distribution:")
        for vtype, count in pd.Series(vehicle_types).value_counts().items():
            print(f"      {vtype:20}: {count:4} ({count/len(vehicle_types)*100:.1f}%)")
        
        return {
            'texts': texts,
            'provinces': provinces,
            'regions': regions,
            'vehicle_types': vehicle_types
        }

    def clean_ocr_noise(self, text):
        """Clean OCR errors"""
        if not isinstance(text, str):
            return ""
        
        text = text.upper().strip().replace(" ", "")
        
        # Parse and clean
        match = re.match(r'^([A-Z]{1,2})([A-Z0-9]{1,5})([A-Z]{0,4})$', text)
        if match:
            p, n, s = match.groups()
            # Clean common OCR errors in number section
            n_clean = n.replace('O', '0').replace('I', '1').replace('L', '1')
            if n_clean.isdigit():
                return f"{p}{n_clean}{s}"
        
        return text

    def parse_plate_text(self, plate_text):
        """
        Parse plate into components (Prefix, Number, Suffix).
        UPDATED: Automatically ignores trailing expiry dates (e.g. '0322').
        """
        clean_plate = self.clean_ocr_noise(plate_text)
        
        # Regex Explanation:
        # ^([A-Z]{1,2}) : Start with 1-2 Letters (Prefix)
        # (\d{1,4})     : Followed by 1-4 Digits (Number)
        # ([A-Z]{0,3})  : Followed by 0-3 Letters (Suffix)
        # We REMOVED the '$' at the end. This allows "0322" to exist but be ignored.
        match = re.match(r'^([A-Z]{1,2})(\d{1,4})([A-Z]{0,3})', clean_plate)
        
        if match:
            return {
                'cleaned': match.group(0), # Only keeps the valid part (B3023KEZ)
                'prefix': match.group(1),  # B
                'number': match.group(2),  # 3023
                'suffix': match.group(3),  # KEZ
                'valid': True
            }
        
        # Fallback: If regex fails, return invalid
        return {
            'cleaned': clean_plate,
            'prefix': '',
            'number': '',
            'suffix': '',
            'valid': False
        }

    def extract_features(self, text):
        """
        Extract features WITHOUT leaking the answer
        NO direct number range indicators
        """
        parsed = self.parse_plate_text(text)
        
        if not parsed['valid']:
            return np.zeros(7)
        
        num = int(parsed['number'])
        
        # Non-leaky features
        features = [
            len(parsed['prefix']),      # Prefix length (1 or 2)
            len(parsed['suffix']),      # Suffix length
            num,                        # Raw number (model learns patterns)
            np.log1p(num),             # Log-scaled number
            num % 1000,                 # Last 3 digits pattern
            1 if parsed['prefix'] == 'B' else 0,  # Jakarta special
            1 if len(parsed['suffix']) == 0 else 0  # No suffix indicator
        ]
        
        return np.array(features)

    def train_and_evaluate(self, test_size=0.2, val_size=0.2):
        """
        Train on manual labels with proper train/val/test split
        """
        if self.training_data is None:
            raise ValueError("No training data loaded!")
        
        data = self.training_data
        
        print(f"\n🚀 Training on {len(data['texts'])} manual labels...")
        
        # 1. Vectorize text
        print("   ↳ Vectorizing text features...")
        self.vectorizer.fit(data['texts'])
        X_tfidf = self.vectorizer.transform(data['texts']).toarray()
        
        # 2. Extract handcrafted features
        print("   ↳ Extracting handcrafted features...")
        X_hand = np.array([self.extract_features(t) for t in data['texts']])
        
        # 3. Combine features
        X = np.hstack([X_tfidf, X_hand])
        
        # 4. Encode labels
        yp = self.province_encoder.fit_transform(data['provinces'])
        yr = self.region_encoder.fit_transform(data['regions'])
        yv = self.vehicle_encoder.fit_transform(data['vehicle_types'])
        
        # 5. Split data: Train (60%), Val (20%), Test (20%)
        print(f"\n✂️  Splitting data: {int((1-test_size-val_size)*100)}% train, {int(val_size*100)}% val, {int(test_size*100)}% test")
        
        # First split: 80% temp, 20% test
        X_temp, X_test, yv_temp, yv_test, yp_temp, yp_test, yr_temp, yr_test = train_test_split(
            X, yv, yp, yr, 
            test_size=test_size, 
            random_state=42, 
            stratify=yv
        )
        
        # Second split: 75% of temp = 60% train, 25% of temp = 20% val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, yv_train, yv_val, yp_train, yp_val, yr_train, yr_val = train_test_split(
            X_temp, yv_temp, yp_temp, yr_temp,
            test_size=val_size_adjusted,
            random_state=42,
            stratify=yv_temp
        )
        
        print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        
        # 6. Train models
        print("\n🏋️  Training classifiers...")
        
        print("   ↳ Province classifier...")
        self.province_classifier.fit(X_train, yp_train)
        
        print("   ↳ Region classifier...")
        self.region_classifier.fit(X_train, yr_train)
        
        print("   ↳ Vehicle classifier...")
        self.vehicle_classifier.fit(X_train, yv_train)
        
        # 7. Evaluate on validation set
        print("\n" + "="*70)
        print("📊 VALIDATION SET RESULTS")
        print("="*70)
        
        self.evaluate_model("Province", self.province_classifier, 
                           self.province_encoder, X_val, yp_val, X_train, yp_train)
        print()
        self.evaluate_model("Region", self.region_classifier, 
                           self.region_encoder, X_val, yr_val, X_train, yr_train)
        print()
        self.evaluate_model("Vehicle Type", self.vehicle_classifier, 
                           self.vehicle_encoder, X_val, yv_val, X_train, yv_train)
        
        # 8. Final evaluation on test set
        print("\n" + "="*70)
        print("📊 TEST SET RESULTS (Final Performance)")
        print("="*70)
        
        results = {}
        results['province'] = self.evaluate_model("Province", self.province_classifier, 
                                                   self.province_encoder, X_test, yp_test)
        print()
        results['region'] = self.evaluate_model("Region", self.region_classifier, 
                                                 self.region_encoder, X_test, yr_test)
        print()
        results['vehicle'] = self.evaluate_model("Vehicle Type", self.vehicle_classifier, 
                                                  self.vehicle_encoder, X_test, yv_test)
        
        return results
    
    def evaluate_model(self, name, model, encoder, X_test, y_test, X_train=None, y_train=None):
        """Evaluate model and check for overfitting"""
        
        # Test accuracy
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        
        # Train accuracy (for overfitting check)
        if X_train is not None and y_train is not None:
            y_train_pred = model.predict(X_train)
            train_acc = accuracy_score(y_train, y_train_pred)
            
            print(f">> {name} Model")
            print(f"   Train Accuracy: {train_acc*100:.2f}%")
            print(f"   Test Accuracy:  {test_acc*100:.2f}%")
            
            # Overfitting check
            gap = train_acc - test_acc
            if gap > 0.15:
                print(f"   ⚠️  OVERFITTING! Gap: {gap*100:.1f}%")
            elif gap > 0.08:
                print(f"   ⚠️  Slight overfitting (gap: {gap*100:.1f}%)")
            else:
                print(f"   ✅ Good generalization (gap: {gap*100:.1f}%)")
        else:
            print(f">> {name} Model - Test Accuracy: {test_acc*100:.2f}%")
        
        # Classification report
        target_names = [str(cls) for cls in encoder.classes_]
        
        # --- THE FIX IS HERE ---
        # We explicitly tell it to expect ALL labels, even if some are missing in X_test
        all_labels = range(len(target_names))
        
        print("\n" + classification_report(
            y_test, 
            y_pred, 
            target_names=target_names, 
            labels=all_labels,  # <--- Added this to prevent crash
            zero_division=0
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix (Rows=Actual, Cols=Predicted):")
        print(cm)
        
        return test_acc



    def predict(self, plate_text, use_fallback=True):
        """
        Predict metadata with HIERARCHICAL LOGIC FIX.
        Prioritizes Rule-based Prefix/Suffix logic over ML when possible.
        """
        parsed = self.parse_plate_text(plate_text)
        
        # 1. Handle Invalid Inputs Immediately
        if not parsed['valid']:
            return {
                'plate': plate_text,
                'vehicle': 'Unknown',
                'region': 'Unknown',
                'province': 'Unknown',
                'confidence': 0.0,
                'status': 'INVALID'
            }
        
        # 2. Run ML Predictions (The "Brain")
        # Prepare features
        X_tfidf = self.vectorizer.transform([parsed['cleaned']]).toarray()
        X_hand = np.array([self.extract_features(parsed['cleaned'])])
        X = np.hstack([X_tfidf, X_hand])
        
        # Predict Vehicle (ML is best at this)
        v_proba = self.vehicle_classifier.predict_proba(X)[0]
        v_idx = v_proba.argmax()
        vehicle = self.vehicle_encoder.inverse_transform([v_idx])[0]
        v_conf = v_proba[v_idx]
        
        # Predict Region & Province (ML guesses)
        r_proba = self.region_classifier.predict_proba(X)[0]
        region = self.region_encoder.inverse_transform([r_proba.argmax()])[0]
        r_conf = r_proba.max()
        
        p_proba = self.province_classifier.predict_proba(X)[0]
        province = self.province_encoder.inverse_transform([p_proba.argmax()])[0]
        p_conf = p_proba.max()
        
        # 3. HIERARCHICAL LOGIC FIX (The "Rule Book")
        # We enforce consistency: If the Prefix is known, IT MUST be correct.
        
        real_prefix = parsed['prefix']
        rule_province = self.province_map.get(real_prefix)
        
        if rule_province:
            # FORCE Province match (Rule > ML)
            # If ML said "Bali" but plate starts with "B", force "Jadetabek"
            if province != rule_province:
                province = rule_province
                p_conf = 1.0 # We are 100% sure because of the prefix rule
        
            # FORCE Region Consistency
            # Case A: Jakarta (B) - Use Suffix Logic
            if real_prefix == 'B' and len(parsed['suffix']) > 0:
                suffix_char = parsed['suffix'][0]
                rule_region = self.jakarta_suffix_map.get(suffix_char)
                if rule_region:
                    region = rule_region
                    r_conf = 0.95 # High confidence in suffix rule
            
            # Case B: Non-Jakarta Conflict Resolution
            # If ML predicts a region that likely belongs to a different island/province
            # (detected by low confidence or mismatch), fallback to General Province name.
            elif real_prefix != 'B':
                # If ML is unsure (<50%) OR the region name implies a known conflict
                # We default the Region to be the same as the Province (General Area)
                if r_conf < 0.5: 
                    region = province  # e.g. Region "Cirebon" instead of "Sidoarjo"
        
        # 4. Calculate Final Confidence
        avg_conf = (v_conf + r_conf + p_conf) / 3
        
        return {
            'plate': parsed['cleaned'],
            'vehicle': vehicle,
            'region': region,
            'province': province,
            'confidence': float(avg_conf),
            'vehicle_confidence': float(v_conf),
            'region_confidence': float(r_conf),
            'province_confidence': float(p_conf),
            'status': 'VALID'
        }

    def save_model(self, filename):
        """Save trained model"""
        model_data = {
            'province_classifier': self.province_classifier,
            'region_classifier': self.region_classifier,
            'vehicle_classifier': self.vehicle_classifier,
            'vectorizer': self.vectorizer,
            'province_encoder': self.province_encoder,
            'region_encoder': self.region_encoder,
            'vehicle_encoder': self.vehicle_encoder,
            'province_map': self.province_map,
            'jakarta_suffix_map': self.jakarta_suffix_map
        }
        joblib.dump(model_data, filename)
        print(f"\n💾 Model saved to: {filename}")

    def load_model(self, filename):
        """Load pre-trained model"""
        model_data = joblib.load(filename)
        
        self.province_classifier = model_data['province_classifier']
        self.region_classifier = model_data['region_classifier']
        self.vehicle_classifier = model_data['vehicle_classifier']
        self.vectorizer = model_data['vectorizer']
        self.province_encoder = model_data['province_encoder']
        self.region_encoder = model_data['region_encoder']
        self.vehicle_encoder = model_data['vehicle_encoder']
        
        print(f"✅ Model loaded from: {filename}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  Indonesian License Plate NLP Extractor")
    print("  Training with Manual Labels Only (No Synthetic)")
    print("=" * 70)
    print()
    
    # 1. Initialize with manual labels
    extractor = ManualLabelsPlateExtractor(
        manual_labels_path='/home/pcsistem/Documents/carlo_ojd_ml/plate_text_dataset/manual_labels.csv'
    )
    
    # 2. Train and evaluate
    results = extractor.train_and_evaluate(test_size=0.15, val_size=0.15)
    
    # 3. Save model
    extractor.save_model('/home/pcsistem/Documents/carlo_ojd_ml/model/nlp_metadata.pkl')
    
    # 4. Test predictions
    print("\n" + "="*70)
    print("🔍 LIVE PREDICTION TESTS")
    print("="*70)
    
    test_plates = [
        'B1234TXY',   # Jakarta
        'E2984TW',    # Cirebon
        'D8888',      # Bandung truck
        'H3520BLG',   # Semarang motorcycle
        'AB1500XY',   # Yogyakarta boundary
        'L6999ABC',   # Surabaya boundary
    ]
    
    for plate in test_plates:
        result = extractor.predict(plate)
        print(f"\nPlate: {result['plate']}")
        print(f"  Province: {result['province']:25} (conf: {result['province_confidence']:.2%})")
        print(f"  Region:   {result['region']:25} (conf: {result['region_confidence']:.2%})")
        print(f"  Vehicle:  {result['vehicle']:25} (conf: {result['vehicle_confidence']:.2%})")
    
    # 5. Stress test with OCR errors
    print("\n" + "="*70)
    print("🧪 STRESS TEST (OCR Errors)")
    print("="*70)
    
    stress_cases = [
        ('B1234TXY', 'Private Car'),      # Clean
        ('BI234TXY', 'Private Car'),      # I->1
        ('B1234TXY', 'Private Car'),      # Clean
        ('D8888', 'Truck/Commercial'),    # Clean truck
        ('D8888', 'Truck/Commercial'),    # Same
        ('H2O2OXY', 'Motorcycle'),        # O->0
        ('L5OOO', 'Motorcycle'),          # O->0
    ]
    
    correct = 0
    for plate, expected in stress_cases:
        result = extractor.predict(plate)
        is_correct = (result['vehicle'] == expected)
        if is_correct:
            correct += 1
        
        status = '✅' if is_correct else '❌'
        print(f"{status} {plate:12} → {result['vehicle']:20} | Expected: {expected}")
    
    print(f"\nStress Test Score: {correct/len(stress_cases)*100:.1f}%")
    
    print("\n" + "="*70)
    print("✅ Training complete! Model saved and ready to use.")
    print("="*70)