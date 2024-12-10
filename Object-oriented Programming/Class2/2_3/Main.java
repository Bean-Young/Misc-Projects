import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.util.Scanner;

public class Main {
    public Main() {
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int year1 = scanner.nextInt();
        int month1 = scanner.nextInt();
        int day1 = scanner.nextInt();
        Date date1 = new Date(year1, month1, day1);
        int year2 = scanner.nextInt();
        int month2 = scanner.nextInt();
        int day2 = scanner.nextInt();
        Date date2 = new Date(year2, month2, day2);
        scanner.close();
        LocalDate localDate1 = date1.toLocalDate();
        LocalDate localDate2 = date2.toLocalDate();
        long days = ChronoUnit.DAYS.between(localDate1, localDate2);
        System.out.println(days);
    }
}
