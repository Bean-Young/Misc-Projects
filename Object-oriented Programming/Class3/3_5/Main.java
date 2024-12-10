import java.text.DecimalFormat;
import java.util.Scanner;

public class Main {
    public Main() {
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int time = scanner.nextInt();
        double price = scanner.nextDouble();
        MobilePhone mobile = new MobilePhone("123456", time, price);
        double fee = mobile.pay();
        DecimalFormat df = new DecimalFormat("#.0");
        String feee = df.format(fee);
        System.out.println("Fee=" + feee);
        scanner.close();
    }
}