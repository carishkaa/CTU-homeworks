import javax.persistence.*;


/**
 * 
 * @author karinabalagazova
 */
@Embeddable
public class Address {
    @Column(name="street")
    private String streetAddress;

    @Column
    private String city;

    @Column(name="zip_code")
    private String areaCode;

    @Override
    public String toString() {
        return streetAddress + ", " + city + ", " + areaCode;
    }
    
    
}
