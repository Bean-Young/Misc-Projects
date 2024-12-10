import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class Main {
    public Main() {
    }

    public static void main(String[] args) {
        String datestr = "2015/07/07";

        try {
            DateFormat df = new SimpleDateFormat("yyyy-MM-dd");
            Date date = df.parse(datestr);
            System.out.println(date);
        } catch (ParseException var8) {
            System.out.println("ERROR-1");
        } catch (Exception var9) {
            System.out.println("ERROR-2");
        } finally {
            datestr = null;
            System.out.println("CLEAR");
        }

    }
}
