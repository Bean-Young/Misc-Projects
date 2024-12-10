import java.time.LocalDate;

class Date {
    int year;
    int month;
    int day;

    public Date(int year, int month, int day) {
        this.year = year;
        this.month = month;
        this.day = day;
    }

    public LocalDate toLocalDate() {
        return LocalDate.of(this.year, this.month, this.day);
    }
}
