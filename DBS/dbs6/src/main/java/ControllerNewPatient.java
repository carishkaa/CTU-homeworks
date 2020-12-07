
import java.io.IOException;
import java.net.URL;
import java.util.ResourceBundle;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;

/**
 * 
 * @author karinabalagazova
 */
public class ControllerNewPatient implements Initializable{
    MyQuery q = MyQuery.getInstance();
    
    @FXML 
    TextField details_name;
    
    @FXML 
    TextField details_birthday;
    
    @FXML 
    TextField details_birthmonth;
    
    @FXML 
    TextField details_birthyear;
    
    @FXML 
    TextField details_blood;
    
    @FXML 
    TextField details_insurance;
    
    @FXML
    Label error_label;
    
    @FXML
    Button saveButton;
    
    //add new patient to Patient database
    @FXML
    private void saveAction(Event event) throws IOException {
        error_label.setText("");
        
        // name
        String name = details_name.getText();
        if (name.matches(".*\\d.*")){
            error_label.setText("name must contain only letters a-z, A-Z");
            return;
        }
        
        // date of birth
        int birthday = Integer.parseInt(details_birthday.getText());
        int birthmonth = Integer.parseInt(details_birthmonth.getText());
        int birthyear = Integer.parseInt(details_birthyear.getText());
        if (birthday > 31 || birthmonth > 12 || birthyear > 2020){
            error_label.setText("please check your date of birth");
            return;
        }
                
        // blood type
        String bloodType = details_blood.getText();
        boolean bloodTypeCheck = bloodType.equalsIgnoreCase("0") || bloodType.equalsIgnoreCase("A") ||
                bloodType.equalsIgnoreCase("B") || bloodType.equalsIgnoreCase("AB");
        if (!bloodTypeCheck){
            error_label.setText("blood type must be 0, A, B or AB");
            return;
        }
        
        int insurance = Integer.parseInt(details_insurance.getText());
        
        q.addNewPatient(name, bloodType, birthday, birthmonth, birthyear, insurance);
        
        saveButton.setDisable(true);
        
        details_name.setEditable(false);
        details_birthday.setEditable(false);
        details_birthmonth.setEditable(false);
        details_birthyear.setEditable(false);
        details_blood.setEditable(false);
        details_insurance.setEditable(false);
        
        details_name.setMouseTransparent(true);
        details_birthday.setMouseTransparent(true);
        details_birthmonth.setMouseTransparent(true);
        details_birthyear.setMouseTransparent(true);
        details_blood.setMouseTransparent(true);
        details_insurance.setMouseTransparent(true);
    }

    @Override
    public void initialize(URL location, ResourceBundle resources) {
    }
}
