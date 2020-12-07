import javax.persistence.*;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

/**
 * 
 * @author karinabalagazova
 */
@Entity
@DiscriminatorValue("doctor")
public class Doctor extends Employee{
    @Column(nullable=false)
    private String department;

    @ManyToMany
    @JoinTable(name="Treatment",
            joinColumns=@JoinColumn(name="employee_id"),
            inverseJoinColumns=@JoinColumn(name="patient_id"))
    private Set<Patient> patients = new HashSet<>();

    
    /**
     * Add relation between the doctor and the given patient
     * @param patient - instance of Patient class
     */
    public void addPatient(Patient patient) {
        patients.add(patient);
        patient.getDoctors().add(this);
    }
    
    /**
     * Remove relation between the doctor and the given patient
     * @param patient - instance of Patient class
     */
    public void removePatient(Patient patient) {
        patients.remove(patient);
        patient.getDoctors().remove(this);
    }
    
    public Collection<Patient> getPatients() {
        return patients;
    }

    public String getDepartment() {
        return department;
    }

    public void setDepartment(String department) {
        this.department = department;
    }
}
