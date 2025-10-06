# StudentHub
# 🎓 Centralised Digital Platform for Comprehensive Student Activity Record in HEIs  
### (Smart Student Hub – Knowledge-Based System)

## 🧠 Project Overview

This project is a **Knowledge-Based System (KBS)** for Higher Education Institutions (HEIs), designed to manage, reason, and verify comprehensive student activity data such as academics, attendance, certificates, and co-curricular achievements.

It acts as a **central digital repository** with intelligent reasoning — enabling admins to define **facts and rules** while students can **query the system**, view insights, and generate verified resumes.

---

## 🚀 Core Features

### 👨‍🏫 Admin Portal
- Secure login with role-based access.  
- Manage student records (add, update, delete).  
- Upload and verify certificates via **OCR (Tesseract)**.  
- Define **rules** (IF–THEN logic) for automated reasoning.  
- View generated inferences such as “At Risk”, “Placement Ready”.  
- Generate reports (NAAC/NBA/NIRF formats).  
- Export results in CSV/PDF format.

### 🎓 Student Portal
- Student login & profile dashboard.  
- Upload activities and certificates (auto OCR verification).  
- View approved and pending achievements.  
- Generate **resume** and **portfolio**.  
- Use built-in **chatbot** for queries (e.g., GPA, attendance, placement readiness).  
- View insights powered by rule-based inference engine.

### 🧩 Knowledge-Based System
- Facts: Student data (GPA, attendance, certifications, activities).  
- Rules: Admin-defined conditions like  
  `IF GPA < 6 AND Attendance < 75 THEN At Risk`.  
- Inference Engine: Evaluates rules over facts to produce actions and recommendations.  
- Safe evaluator prevents arbitrary code execution.

### 🧾 Certificate Verification (OCR)
- Uses **Tesseract OCR** with OpenCV preprocessing.  
- Extracts text fields like name, course, date, and organization.  
- Fuzzy matches student details to verify authenticity.  
- Assigns verification score and flags inconsistencies.

### 💬 Chatbot
- Rule-based, context-aware chatbot for students.  
- Handles queries about GPA, CGPA, attendance, activities, or placement readiness.  
- Fallbacks to rule-based inference summaries.  
- (Future scope: integrate NLP model for natural conversation.)

### 📈 Reports & Analytics
- Department-wise statistics and dashboards.  
- Export-ready NAAC/NBA/NIRF data.  
- Visual analytics for academic trends and participation.
