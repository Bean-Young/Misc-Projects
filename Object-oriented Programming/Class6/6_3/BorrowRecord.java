import java.time.LocalDate;
import java.time.temporal.ChronoUnit;

class BorrowRecord {
    String id;
    String user;
    String bookName;
    LocalDate borrowDate;
    LocalDate dueDate;
    double fee;
    String status;
    double price;
    int pages;

    public BorrowRecord(String id, String user, String bookName, LocalDate borrowDate) {
        this.id = id;
        this.user = user;
        this.bookName = bookName;
        this.borrowDate = borrowDate;
        this.dueDate = borrowDate.plusDays(14L);
        this.status = "未归还";
    }

    public void returnBook(LocalDate returnDate) {
        if (returnDate.isAfter(this.dueDate)) {
            long daysLate = ChronoUnit.DAYS.between(this.dueDate, returnDate);
            this.fee += (double)daysLate * 0.1;
        }

        this.status = "已归还";
    }

    public String toString() {
        return "BorrowRecord{id='" + this.id + "', user='" + this.user + "', bookName='" + this.bookName + "', borrowDate=" + this.borrowDate + ", dueDate=" + this.dueDate + ", fee=" + this.fee + ", status='" + this.status + "'}";
    }
}
