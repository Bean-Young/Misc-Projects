import java.io.PrintStream;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;

public class Main {
    public Main() {
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd");
        List<BorrowRecord> records = new ArrayList();
        String input = scanner.nextLine();
        String[] commands = input.split(" ");
        int i = 0;

        while(true) {
            while(i < commands.length) {
                switch (commands[i]) {
                    case "borrow":
                        String id = commands[i + 1];
                        String user = commands[i + 2];
                        String bookName = commands[i + 3];
                        String borrowDateStr = commands[i + 4];
                        LocalDate borrowDate = LocalDate.parse(borrowDateStr, formatter);
                        BorrowRecord record = new BorrowRecord(id, user, bookName, borrowDate);
                        records.add(record);
                        PrintStream var24 = System.out;
                        String var10001 = record.dueDate.format(formatter);
                        var24.print("借阅成功，预计归还日期：" + var10001 + "。");
                        i += 5;
                        break;
                    case "return":
                        String returnId = commands[i + 1];
                        String returnDateStr = commands[i + 2];
                        LocalDate returnDate = LocalDate.parse(returnDateStr, formatter);
                        Iterator var25 = records.iterator();

                        while(var25.hasNext()) {
                            BorrowRecord rec = (BorrowRecord)var25.next();
                            if (rec.id.equals(returnId)) {
                                rec.returnBook(returnDate);
                                String formattedFee = String.format("%.2f", rec.fee).replaceAll("0*$", "").replaceAll("\\.$", "");
                                System.out.print("书籍已归还，额外费用：" + formattedFee + "元。");
                                break;
                            }
                        }

                        i += 3;
                        break;
                    case "damage":
                        int flag = 0;
                        String var10000 = commands[i + 1];
                        var10000 = commands[i + 2];
                        String damageBookName = commands[i + 3];
                        if (damageBookName.equals("普通书刊")) {
                            System.out.print("图书报损完成，赔偿金额为书籍价格。");
                        } else if (damageBookName.equals("教师讲义")) {
                            String damagePrice = commands[i + 4];
                            flag = 1;
                            System.out.print("图书报损完成，赔偿金额为" + damagePrice + "元。");
                        }

                        i += 4 + flag;
                        break;
                    default:
                        ++i;
                }
            }

            scanner.close();
            return;
        }
    }
}
