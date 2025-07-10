import sqlite3
from datetime import date

def insert_sample_data():
    conn = sqlite3.connect("FinalProject.db")
    c = conn.cursor()

    # Insert sample employers
    employers = [
        ("Tech Solutions Inc", "Leading tech company with innovative solutions", "Technology", "100-500"),
        ("Healthcare Plus", "Modern healthcare provider", "Healthcare", "500-1000"),
        ("Global Finance", "International financial services", "Finance", "1000+")
    ]

    for company_name, desc, industry, size in employers:
        try:
            # First check if employer exists
            existing = c.execute("SELECT user_id FROM Users WHERE email = ?", 
                               (f"admin@{company_name.lower().replace(" ", "")}.com",)).fetchone()
            if existing:
                continue

            c.execute("""
                INSERT INTO Users (first_name, last_name, email, password, user_type)
                VALUES (?, ?, ?, ?, ?)
            """, (company_name, "Admin", f"admin@{company_name.lower().replace(" ", "")}.com", "password123", "employer"))
            
            employer_id = c.lastrowid
            
            c.execute("""
                INSERT INTO Employer (user_id, company_name, company_description, industry, company_size)
                VALUES (?, ?, ?, ?, ?)
            """, (employer_id, company_name, desc, industry, size))

            # Insert jobs for this employer
            jobs = [
                ("Senior Software Engineer", "Expert in Python development", 120000),
                ("Data Scientist", "ML and AI experience required", 100000),
                ("Product Manager", "Lead technical initiatives", 110000)
            ]
            
            for title, desc, salary in jobs:
                c.execute("""
                    INSERT INTO Jobs (employer_id, title, description, salary, location, date_posted, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (employer_id, title, desc, salary, "Remote", date.today(), "open"))

        except sqlite3.IntegrityError:
            print(f"Skipping duplicate entry for {company_name}")
            continue

    # Insert sample job seekers
    seekers = [
        ("John", "Doe", "john@example.com", "Python, JavaScript", "3 years experience"),
        ("Jane", "Smith", "jane@example.com", "Data Science, ML", "2 years experience")
    ]

    for fname, lname, email, skills, experience in seekers:
        try:
            # Check if user exists
            existing = c.execute("SELECT user_id FROM Users WHERE email = ?", (email,)).fetchone()
            if existing:
                continue

            c.execute("""
                INSERT INTO Users (first_name, last_name, email, password, user_type)
                VALUES (?, ?, ?, ?, ?)
            """, (fname, lname, email, "password123", "job_seeker"))
            
            seeker_id = c.lastrowid
            
            c.execute("""
                INSERT INTO JobSeeker (user_id, skills, experience)
                VALUES (?, ?, ?)
            """, (seeker_id, skills, experience))

        except sqlite3.IntegrityError:
            print(f"Skipping duplicate entry for {email}")
            continue

    conn.commit()
    conn.close()
    print("Sample data inserted successfully!")

if __name__ == "__main__":
    insert_sample_data()
