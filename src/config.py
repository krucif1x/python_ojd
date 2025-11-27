# config.py

# Province and region mapping
PROVINCE_MAP = {
    "A": "Banten", "B": "DKI Jakarta", "D": "Jawa Barat", 
    "F": "Bogor", "E": "Cirebon", "G": "Pekalongan",
    "H": "Semarang", "K": "Pati", "L": "Surabaya",
    "N": "Malang", "W": "Sidoarjo",
}

# Region mapping for PREFIX (first 2-3 letters)
REGION_PREFIX_MAP = {
    "AB": "Sleman", "AD": "Surakarta", "AE": "Madiun",
    "AG": "Kediri", "BA": "Lampung", "DA": "Banjarmasin",
    "KB": "Pontianak",
}

# Vehicle type mapping (if you still want basic vehicle type without color)
VEHICLE_TYPE_MAP = {
    "default": "Private Vehicle",  # Default type when color is not analyzed
}

# Model and image settings
MODEL_IMG_SIZE = (40, 40)
CLASS_NAMES = ['0','1','2','3','4','5','6','7','8','9'] + [chr(ord('A')+i) for i in range(26)]

# Pipeline settings
DEFAULT_RESIZE_SCALE = 0.4
DEFAULT_PLATE_MIN_WIDTH = 200
DEFAULT_PLATE_MAX_ASPECT = 4
DEFAULT_CHAR_MIN_H = 40
DEFAULT_CHAR_MAX_H = 200
DEFAULT_CHAR_MIN_W = 8