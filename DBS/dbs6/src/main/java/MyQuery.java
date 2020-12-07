import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.persistence.*;

/**
 * 
 * @author karinabalagazova
 */
public class MyQuery {
    private static MyQuery query = null;
    private final EntityManagerFactory emf;
    private final EntityManager em;
    private final EntityTransaction et;
    private Controller c;

    public MyQuery() {
        emf = Persistence.createEntityManagerFactory("ApplicationPU");
        em = emf.createEntityManager();
        et = em.getTransaction();
    }

    /**
     * Create instance of MyQuery Class if it isn't exist and returns it
     * @return MyQuery Class instance
     */
    public static MyQuery getInstance(){
        if (query == null){
            query = new MyQuery();
        }
        return query;
    }
    
    public EntityManager getEntityManager(){
        return em;
    }

    public void beginTransaction(){
        et.begin();
    }

    public void commitTransaction(){
        et.commit();
    }
    
    /**
     * Deletes patient with the given id
     * @param id - patient id
     * @return true if successful
     */
    public boolean deletePatient(String id){
        beginTransaction();
        int count = em.createQuery("DELETE FROM Patient p " 
                + "WHERE p.patient_id = " + id).executeUpdate();
        
        commitTransaction();
        return count==1;
    }
    
    /**
     * Update information about patient 
     * @param id - patient's id 
     * @param name - updated name
     * @param bloodType - updated blood type
     * @param insurance - updated insurance id
     */
    public void update(int id, String name, String bloodType, int insurance){
        beginTransaction();
        Patient patient= (Patient) em.find (Patient.class, id);
        patient.setName(name);
        patient.setBloodType(bloodType);
        patient.setInsurance_id(insurance);
        commitTransaction();
    }
    
    /**
     * Get Patient class instance using patient id
     * @param id - patient id
     * @return Patient instance
     */
    public Patient findPatient(int id) {
        beginTransaction();
        Patient patient = (Patient) em.find (Patient.class, id);
        commitTransaction();
        return patient;
    }
    
    /**
     * Get Doctor class instance using doctor id
     * @param id - doctor id
     * @return Doctor instance
     */
    public Doctor findDoctor(int id) {
        beginTransaction();
        Doctor d = (Doctor) em.find (Doctor.class, id);
        commitTransaction();
        return d;
    }
    
    /**
     * Adds new patient to the database
     * @param name
     * @param bloodType
     * @param day
     * @param month
     * @param year
     * @param insurance 
     */
    public void addNewPatient(String name, String bloodType, int day, int month, int year, int insurance){
        beginTransaction();
        
        Patient patient = new Patient();
        patient.setName(name);
        patient.setDateOfBirth(new Date(year-1900, month-1, day));
        patient.setBloodType(bloodType);
        patient.setInsurance_id(insurance);
        
        em.persist(patient);
        System.out.println("add new" + patient);
        commitTransaction();
    }
    
    /**
     * Get information about all patients
     * @return map of lists with information
     */
    public Map<String,List<String>> readPatientQuery(){
       
        List<String> ids = new ArrayList<>();
        List<String> names = new ArrayList<>();
        List<String> dateOfBirth = new ArrayList<>();
        List<String> bloodType = new ArrayList<>();
        List<String> insurance = new ArrayList<>();
        
        TypedQuery<Patient> tq = em.createQuery("SELECT p FROM Patient p", Patient.class);
            for (Patient p: tq.getResultList()){
                // id
                ids.add(p.getID().toString());
                
                //name
                names.add(p.getName());
                
                //date of birth
                SimpleDateFormat formatter = new SimpleDateFormat("dd/MM/yyyy");  
                String strDate = formatter.format(p.getDateOfBirth());  
                dateOfBirth.add(strDate);
                
                //blood type
                bloodType.add(p.getBloodType());
                
                //insurance
                insurance.add(p.getInsurance_id().toString());
                
                System.out.println("id = " + p.getID() + " | name = " + p.getName() + " | " + p.getDateOfBirth());
            }
        Map<String,List<String>> map = new HashMap();
        
        map.put("id", ids);
        map.put("name", names);
        map.put("date of birth", dateOfBirth);
        map.put("blood type", bloodType);
        map.put("insurance", insurance);
        
        System.out.println(map);
        return map;
    }
    
    /**
     * Get information about all doctors
     * @return map of lists with information
     */
    public Map<String,List<String>> readDoctorQuery(){
       
        List<String> ids = new ArrayList<>();
        List<String> names = new ArrayList<>();
        List<String> dateOfBirth = new ArrayList<>();
        List<String> positions = new ArrayList<>();
        List<String> address = new ArrayList<>();
        
        TypedQuery<Doctor> tq = em.createQuery("SELECT p FROM Doctor p", Doctor.class);
            for (Doctor p: tq.getResultList()){
                // id
                ids.add(p.getID().toString());
                
                //name
                names.add(p.getName());
                
                //date of birth
                SimpleDateFormat formatter = new SimpleDateFormat("MM/dd/yyyy");  
                String strDate = formatter.format(p.getDateOfBirth());  
                dateOfBirth.add(strDate);
                
                //blood type
                positions.add(p.getPosition());
                
                //insurance
                address.add(p.getAddress().toString());
                
                System.out.println("id = " + p.getID() + " | name = " + p.getName() + " | " + p.getDateOfBirth());
            }
        Map<String,List<String>> map = new HashMap();
        
        map.put("id", ids);
        map.put("name", names);
        map.put("date of birth", dateOfBirth);
        map.put("position", positions);
        map.put("address", address);
        
        System.out.println(map);
        return map;
    }
    
    
}
