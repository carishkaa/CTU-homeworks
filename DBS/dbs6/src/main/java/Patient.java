import javax.persistence.*;
import java.util.Collection;
import java.util.Date;
import java.util.HashSet;
import java.util.Set;

/**
 * 
 * @author karinabalagazova
 */
@Entity
public class Patient {
    @Id
    @GeneratedValue(strategy = GenerationType.SEQUENCE)
    private Integer patient_id;

    @Column(name="patient_name", nullable=false)
    private String name;

    @Temporal(TemporalType.DATE)
    private Date dateOfBirth;

    @Column(nullable=false)
    private String bloodType;

    @Column(unique=true, nullable=false)
    private Integer insurance_id;

    @ManyToMany(mappedBy = "patients")
    private Set<Doctor> doctors = new HashSet<>();

    public Integer getID() {
        return patient_id;
    }

    public void setID(Integer patient_id) {
        this.patient_id = patient_id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Date getDateOfBirth() {
        return dateOfBirth;
    }

    public void setDateOfBirth(Date dateOfBirth) {
        this.dateOfBirth = dateOfBirth;
    }

    public String getBloodType() {
        return bloodType;
    }

    public void setBloodType(String bloodType) {
        this.bloodType = bloodType;
    }

    public Integer getInsurance_id() {
        return insurance_id;
    }

    public void setInsurance_id(Integer insurance_id) {
        this.insurance_id = insurance_id;
    }

    public Collection<Doctor> getDoctors() {
        return doctors;
    }

    public void setDoctors(Set<Doctor> doctors) {
        this.doctors = doctors;
    }
    
}