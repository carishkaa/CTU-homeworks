import javax.persistence.*;
import java.util.Date;

/**
 * 
 * @author karinabalagazova
 */
@Entity
@Table(uniqueConstraints= @UniqueConstraint(columnNames={"name", "dateOfBirth"}))
@Inheritance(strategy = InheritanceType.JOINED)
@DiscriminatorColumn(name="employee_type", discriminatorType = DiscriminatorType.STRING)
public class Employee {
    @Id
    @GeneratedValue
    private Integer employee_id;

    @Column
    private String name;

    @Column
    private String position;

    @Temporal(TemporalType.DATE)
    private Date dateOfBirth;

    @Embedded
    private Address address;

    public Integer getID() {
        return employee_id;
    }

    public void setID(Integer employee_id) {
        this.employee_id = employee_id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getPosition() {
        return position;
    }

    public void setPosition(String position) {
        this.position = position;
    }

    public Date getDateOfBirth() {
        return dateOfBirth;
    }

    public void setDateOfBirth(Date dateOfBirth) {
        this.dateOfBirth = dateOfBirth;
    }

    public Address getAddress() {
        return address;
    }

    public void setAddress(Address address) {
        this.address = address;
    }
}
