import java.io.IOException;
import java.net.URL;
import java.util.List;
import java.util.Map;
import java.util.ResourceBundle;
import java.util.Set;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.ListView;
import javafx.stage.Stage;

/**
 * 
 * @author karinabalagazova
 */
public class Controller implements Initializable{
    MyQuery q = MyQuery.getInstance();
    
    private Map<String,List<String>> patient_list;
    private Map<String,List<String>> doctor_list;
    
    @FXML
    private ListView<String> patients = new ListView<String>();
    
    @FXML
    private ListView<String> doctors = new ListView<String>();
    
    @FXML
    private Button detailsPatientButton;
    
    @FXML
    private Button deletePatientButton;
    
    @FXML
    private Button addNewPatientButton;
    
    @FXML
    private Button addRelationButton;
    
    @FXML
    private Button removeRelationButton;
    
    /**
     * Shows a list of patients on the main window
     * @param patientsList 
     */
    public void setPatientsListView(ObservableList<String> patientsList) {
        patients.setItems(patientsList);
        patients.refresh();
    }
    
    /**
     * Shows a list of doctors on the main window
     * @param doctorsList 
     */
    public void setDoctorsListView(ObservableList<String> doctorsList) {
        doctors.setItems(doctorsList);
        doctors.refresh();
    }
    
    
    private void refresh(){
        patient_list = q.readPatientQuery();
        List<String> pnames = patient_list.get("name");
        ObservableList<String> patientListView = FXCollections.observableArrayList(pnames);
        this.setPatientsListView(patientListView);
        
        doctor_list = q.readDoctorQuery();
        List<String> dnames = doctor_list.get("name");
        ObservableList<String> doctorListView = FXCollections.observableArrayList(dnames);
        this.setDoctorsListView(doctorListView);
    }
    
    @FXML
    private void addNewPatientAction(Event event) throws IOException {
        FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("/new_patient.fxml"));
        Parent root = (Parent) fxmlLoader.load();
                
        Stage stage = new Stage();
        stage.setScene(new Scene(root));
        stage.showAndWait();
        refresh();
    }
    
    @FXML
    private void deletePatientAction(Event event) throws IOException {
        int index = patients.getSelectionModel().getSelectedIndex();
        Integer patient_id = Integer.parseInt(patient_list.get("id").get(index));
        Patient patient = q.findPatient(patient_id);
        
        Set<Doctor> doctors;
        for (Doctor d : patient.getDoctors()){
            d.removePatient(patient);
        }
        q.deletePatient(patient_id.toString());
        refresh();
    }
    
    @FXML
    private void addRelationAction(Event event) throws IOException {
        //get patient
        int index = patients.getSelectionModel().getSelectedIndex();
        int patient_id = Integer.parseInt(patient_list.get("id").get(index));
        Patient patient = q.findPatient(patient_id);
        
        // get doctor
        index = doctors.getSelectionModel().getSelectedIndex();
        int doctor_id = Integer.parseInt(doctor_list.get("id").get(index));
        Doctor doctor = q.findDoctor(doctor_id);
        
        // add relation
        doctor.addPatient(patient);
    }
    
    @FXML
    private void removeRelationAction(Event event) throws IOException {
        // get patient
        int index = patients.getSelectionModel().getSelectedIndex();
        int patient_id = Integer.parseInt(patient_list.get("id").get(index));
        Patient patient = q.findPatient(patient_id);
        
        // get doctor
        index = doctors.getSelectionModel().getSelectedIndex();
        int doctor_id = Integer.parseInt(doctor_list.get("id").get(index));
        Doctor doctor = q.findDoctor(doctor_id);
        
        // remove relation
        doctor.removePatient(patient);
    }
    
    @FXML
    private void refreshAction(Event event) throws IOException {
        this.refresh();
    }
    
    @FXML
    private void viewPatientDetailsAction(Event event) throws IOException {
        FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("/details_patient.fxml"));
        Parent root = (Parent) fxmlLoader.load();
        
        ControllerPatientDetails c = fxmlLoader.getController();
        int index = patients.getSelectionModel().getSelectedIndex();
        c.printDetails(index, patient_list);
        
        Stage stage = new Stage();
        stage.setScene(new Scene(root));
        stage.show();
    }
    
    @FXML
    private void viewDoctorDetailsAction(Event event) throws IOException {
        FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("/details_doctor.fxml"));
        Parent root = (Parent) fxmlLoader.load();
        
        ControllerDoctorDetails c = fxmlLoader.getController();
        int index = doctors.getSelectionModel().getSelectedIndex();
        if (index == -1)
            return;
        
        c.printDetails(index, doctor_list);
        Stage stage = new Stage();
        stage.setScene(new Scene(root));
        stage.show();
    }

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        // view list of all patients
        patient_list = q.readPatientQuery();
        List<String> pnames = patient_list.get("name");
        ObservableList<String> patientListView = FXCollections.observableArrayList(pnames);
        this.setPatientsListView(patientListView);
        
        // view list of all doctors
        doctor_list = q.readDoctorQuery();
        List<String> dnames = doctor_list.get("name");
        ObservableList<String> doctorListView = FXCollections.observableArrayList(dnames);
        this.setDoctorsListView(doctorListView);
    }
}
