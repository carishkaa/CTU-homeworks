
-- pacienti s nespavosti, kteri tento rok nenavstivili doktora
	SELECT patient_id, patient_name
	FROM Patient 
	INNER JOIN Diagnosing USING (patient_id)
	INNER JOIN Diagnosis USING (diagnosis_id)
	WHERE (diagnosis_name = 'Insomnia')
EXCEPT
	SELECT DISTINCT patient_id, patient_name
	FROM Patient 
	INNER JOIN Examination USING (patient_id)
	WHERE (examination_date >= '2020-01-01');

-- seznam diagnoz, ktere byly nalezeny u vice nez 10 lidi (od nejpopularnejsich (DESC))
SELECT diagnosis_name, COUNT(*)
FROM Diagnosis LEFT OUTER JOIN Diagnosing USING(diagnosis_id)
GROUP BY diagnosis_name
HAVING (COUNT(*) > 10)
ORDER BY COUNT(*) DESC;

-- jmena lekaru, kteri provedli vysetreni v mistnosti 323 za pritomnosti zdravotni sestry
SELECT Employee.name
FROM Employee
NATURAL JOIN(
	SELECT DISTINCT doctor AS employee_id
	FROM Examination
	WHERE (nurse IS NOT NULL) AND (examination_room = 323)
) AS D;

-- pacienti, kteri alespon jednou byli vysetreni lekarem z oddeleni kardiologie
SELECT DISTINCT patient_id, patient_name 
FROM Patient
JOIN Examination USING (patient_id)
JOIN Doctor ON (Examination.doctor = Doctor.employee_id)
JOIN Department ON (Doctor.department = Department.code)
WHERE (Department.name = 'Cardiology');

-- pocet dostupnych jednoluzkovych pokoju v kazdem bloku
SELECT block, COUNT(*) FROM Room
WHERE (type = 'Single') AND (available IS TRUE)
GROUP BY block
ORDER BY block;