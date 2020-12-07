
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.ResourceBundle;
import javafx.collections.FXCollections;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ListView;
import javafx.scene.control.TextField;

/**
 * 
 * @author karinabalagazova
 */
public class ControllerPatientDetails implements Initializable{
    MyQuery q = MyQuery.getInstance();
    
    @FXML 
    TextField details_name;
    
    @FXML 
    TextField details_birth;
    
    @FXML 
    TextField details_blood;
    
    @FXML 
    TextField details_insurance;
    
    @FXML
    Label id_label;
    
    @FXML 
    Button editButton;
    
    @FXML
    private ListView<String> doctorsListView = new ListView<String>();
    
    
    /**
     * Print information about the given patient 
     * @param index - index of given patient in the list of the main window
     * @param patient_list - map of all patients' information
     */
    public void printDetails(Integer index, Map<String,List<String>> patient_list){
        String id = patient_list.get("id").get(index);
        id_label.setText(id);
        details_name.setText(patient_list.get("name").get(index));
        details_birth.setText(patient_list.get("date of birth").get(index));
        details_blood.setText(patient_list.get("blood type").get(index));
        details_insurance.setText(patient_list.get("insurance").get(index));
        
        // doctors list
        Patient patient = q.findPatient(Integer.parseInt(id));
        Collection<Doctor> doctors = patient.getDoctors();
        List<String> doctor_names = new ArrayList<>();
        for (Doctor d : doctors){
            doctor_names.add(d.getName());
        }
        doctorsListView.setItems(FXCollections.observableArrayList(doctor_names));
        doctorsListView.refresh();
    }
    
    private static boolean flag = true;
    // edit patient
    @FXML
    private void editAction(Event event) throws IOException {
        details_name.setEditable(true);
        details_blood.setEditable(flag);
        details_insurance.setEditable(flag);
        
        details_name.setMouseTransparent(!flag);
        details_blood.setMouseTransparent(!flag);
        details_insurance.setMouseTransparent(!flag);
        
        if (flag){
            editButton.setText("OK");
        } else {
            q.update(Integer.parseInt(id_label.getText()), details_name.getText(), details_blood.getText(), Integer.parseInt(details_insurance.getText()));
            editButton.setText("Edit");
        }
        
        flag = !flag;
    }

    @Override
    public void initialize(URL location, ResourceBundle resources) {
    }
    
}
