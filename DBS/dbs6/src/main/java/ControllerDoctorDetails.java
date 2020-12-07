import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import javafx.collections.FXCollections;
import javafx.fxml.FXML;
import javafx.scene.control.Label;
import javafx.scene.control.ListView;
import javafx.scene.control.TextField;

/**
 * 
 * @author karinabalagazova
 */
public class ControllerDoctorDetails{
    MyQuery q = MyQuery.getInstance();
    
    @FXML 
    TextField details_name;
    
    @FXML 
    TextField details_birth;
    
    @FXML 
    TextField details_position;
    
    @FXML 
    TextField details_address;
    
    @FXML
    Label id_label;
    
    @FXML
    private ListView<String> patientsListView = new ListView<String>();
    
    /**
     * Print information about the given doctor 
     * @param index - index of given doctor in the list of the main window
     * @param doctor_list - map of all doctors' information
     */
    public void printDetails(Integer index, Map<String,List<String>> doctor_list){
        String id = doctor_list.get("id").get(index);
        id_label.setText(id);
        details_name.setText(doctor_list.get("name").get(index));
        details_birth.setText(doctor_list.get("date of birth").get(index));
        details_position.setText(doctor_list.get("position").get(index));
        details_address.setText(doctor_list.get("address").get(index));
        
        // get doctor instance
        Doctor doctor = q.findDoctor(Integer.parseInt(id));
        
        // view list of current patient of the doctor
        Collection<Patient> patients = doctor.getPatients();
        List<String> patient_names = new ArrayList<>();
        for (Patient p : patients){
            patient_names.add(p.getName());
        }
        patientsListView.setItems(FXCollections.observableArrayList(patient_names));
        patientsListView.refresh();
    }

}
