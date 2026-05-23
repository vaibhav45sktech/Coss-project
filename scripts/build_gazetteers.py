"""
Build Indian-context PII gazetteers for the rule-based detector.

Outputs:
  data/gazetteers/indian_names.txt    — ~200 first names, lowercase, sorted
  data/gazetteers/indian_villages.txt — ~150 districts/towns, lowercase, sorted

These feed IndianPIIRules.detect_names and detect_villages. The lists aim
for regional diversity (North/South/East/West, multi-religion) so the
gazetteer reflects realistic Indian agricultural user base rather than
a single-region bias.

Run from repo root:
    python scripts/build_gazetteers.py
"""

from pathlib import Path

NAMES = [
    # North Indian Hindu
    "ramesh", "suresh", "rajesh", "mukesh", "dinesh", "naresh", "hitesh",
    "amit", "rohit", "rohan", "ravi", "rakesh", "manish", "ashish", "ankit",
    "vikram", "vikas", "sanjay", "sandeep", "saurabh", "shubham", "siddharth",
    "priya", "pooja", "neha", "sunita", "kavita", "anita", "meera", "rita",
    "nisha", "deepika", "shweta", "rekha", "anjali", "preeti", "seema",
    # South Indian (Karnataka, Tamil Nadu, Andhra, Telangana, Kerala)
    "lakshmi", "arjun", "kavya", "karthik", "divya", "murali", "saraswati",
    "venkat", "padma", "srinivas", "sita", "selvam", "kavitha", "mageshwari",
    "senthil", "murugan", "selvi", "bharathi", "ganesan", "lavanya", "sneha",
    "nagarjuna", "padmavathi", "bhaskar", "anjaneya", "ramya", "vidya",
    "harish", "girish", "ganesh", "mahesh", "ashok", "prakash", "shankar",
    "krishna", "raju", "naveen", "praveen", "bharath", "vijay", "ajay",
    # Muslim names (across regions)
    "ahmed", "mohammed", "imran", "faisal", "salim", "rashid", "akbar",
    "yusuf", "ibrahim", "khalid", "tariq", "sameer", "javed", "irfan",
    "fatima", "aisha", "zeenat", "saira", "nasreen", "yasmin", "rehana",
    "shabana", "naseema", "farida", "noor", "tabassum",
    # Sikh / Punjabi
    "harpreet", "jaspreet", "manpreet", "gurpreet", "ranjit", "simran",
    "davinder", "amrit", "baldev", "harjit", "sukhwinder", "kuldeep",
    "balwinder", "rajwinder", "parminder", "satinder",
    # Christian
    "john", "mary", "joseph", "susan", "thomas", "anna", "daniel", "grace",
    "peter", "paul", "george", "elizabeth", "stephen", "rebecca",
    # Bengali / Eastern
    "subir", "rita", "soumya", "mitali", "anirban", "tapan", "mou",
    "debashish", "ranjan", "shyamali", "amitabh", "bidisha",
    # Marathi / Western
    "aditya", "vivek", "pranav", "madhuri", "yogesh", "smita", "rohini",
    "sachin", "milind", "sushma", "vaishali",
    # Gujarati
    "kiran", "hardik", "jignesh", "bhavna", "nilesh", "paresh", "darshan",
    "hemant", "jayesh", "alpa",
    # Odia / Assamese
    "biswajit", "manas", "lipika", "abinash", "nayan", "dhruba",
    # Tribal / regional additions
    "birsa", "phulmoni", "kanha", "lalita",
    # Names that appear in your synthetic data
    "heer", "vaibhav", "abhishek", "abhilash", "aarti", "alok",
    # Common short forms
    "raj", "kumar", "singh", "devi", "rao", "reddy",
    # Names from observed synthetic data + commonly missed
    "navya", "shaurya", "ishita", "tanvi", "myra", "aanya", "kiara",
    "advait", "reyansh", "atharv", "vihaan", "shaurya", "kabir",
    "tara", "diya", "saanvi", "anika", "khushi", "ananya",
    "lakshay", "ayaan", "rudra", "viraj", "neel", "om",
]

VILLAGES = [
    # Karnataka
    "kolar", "mysuru", "hassan", "belagavi", "tumkur", "mandya", "hubli",
    "dharwad", "bellary", "bidar", "raichur", "shimoga", "chitradurga",
    "chamarajanagar", "kodagu", "udupi", "mangaluru", "gadag", "haveri",
    # Andhra Pradesh / Telangana
    "anantapur", "guntur", "vijayawada", "tirupati", "chittoor", "kakinada",
    "rajahmundry", "warangal", "karimnagar", "nizamabad", "adilabad",
    "khammam", "nalgonda", "kurnool", "nellore", "ongole", "eluru",
    # Tamil Nadu
    "coimbatore", "salem", "vellore", "erode", "tirunelveli", "madurai",
    "tiruchirappalli", "thanjavur", "dindigul", "karur", "namakkal",
    "krishnagiri", "dharmapuri", "kanchipuram", "cuddalore",
    # Maharashtra
    "nashik", "kolhapur", "pune", "aurangabad", "solapur", "sangli",
    "satara", "ahmednagar", "jalgaon", "akola", "amravati", "nanded",
    "latur", "wardha", "yavatmal", "buldhana",
    # Punjab / Haryana
    "ludhiana", "amritsar", "jalandhar", "patiala", "bathinda", "hoshiarpur",
    "moga", "firozpur", "faridkot", "sangrur", "barnala",
    "hisar", "karnal", "sonipat", "rohtak", "ambala", "kurukshetra",
    "panipat", "kaithal", "jind",
    # Uttar Pradesh / Bihar
    "meerut", "agra", "bareilly", "aligarh", "mathura", "lucknow",
    "kanpur", "varanasi", "gorakhpur", "moradabad", "saharanpur",
    "muzaffarnagar", "muzaffarpur", "patna", "gaya", "bhagalpur",
    # Madhya Pradesh / Rajasthan
    "indore", "bhopal", "gwalior", "jabalpur", "ujjain", "ratlam",
    "jaipur", "jodhpur", "kota", "ajmer", "bikaner", "alwar", "bharatpur",
    # Gujarat
    "ahmedabad", "surat", "vadodara", "rajkot", "junagadh", "anand",
    "bhavnagar", "jamnagar", "navsari",
    # West Bengal / Odisha
    "kolkata", "howrah", "siliguri", "durgapur", "asansol", "kharagpur",
    "bhubaneswar", "cuttack", "puri", "balasore", "sambalpur",
    # Common name suffixes (often appear as standalone-ish)
    "nagar", "puram", "halli", "patti", "pur", "gaon", "wada", "palli",
    "kheda",
]


def main() -> None:
    out_dir = Path("data/gazetteers")
    out_dir.mkdir(parents=True, exist_ok=True)

    names_clean = sorted({n.strip().lower() for n in NAMES if n.strip()})
    villages_clean = sorted({v.strip().lower() for v in VILLAGES if v.strip()})

    (out_dir / "indian_names.txt").write_text("\n".join(names_clean) + "\n", encoding="utf-8")
    (out_dir / "indian_villages.txt").write_text("\n".join(villages_clean) + "\n", encoding="utf-8")

    print(f"Wrote {len(names_clean)} names -> {out_dir / 'indian_names.txt'}")
    print(f"Wrote {len(villages_clean)} villages -> {out_dir / 'indian_villages.txt'}")


if __name__ == "__main__":
    main()
