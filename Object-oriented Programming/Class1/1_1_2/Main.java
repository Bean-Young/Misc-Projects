import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;

public class Main {
    public Main() {
    }

    public static void main(String[] args) {
        List<Student> students = new ArrayList();
        Scanner scanner = new Scanner(System.in);
        int numberOfStudents = Integer.parseInt(scanner.nextLine());

        for(int i = 0; i < numberOfStudents; ++i) {
            String input = scanner.nextLine();
            String[] parts = input.split(" ");
            String name = parts[0];
            int score = Integer.parseInt(parts[1]);
            students.add(new Student(name, score));
        }

        students.sort((s1, s2) -> {
            return s2.score - s1.score;
        });
        Iterator var9 = students.iterator();

        while(var9.hasNext()) {
            Student s = (Student)var9.next();
            printStudent(s);
        }

    }

    private static void printStudent(Student student) {
        int i;
        for(i = 0; i < 15 - student.name.length(); ++i) {
            System.out.print(" ");
        }

        System.out.print(student.name);

        for(i = 0; i < 5 - String.valueOf(student.score).length(); ++i) {
            System.out.print(" ");
        }

        System.out.println(student.score);
    }

    static class Student {
        String name;
        int score;

        public Student(String name, int score) {
            this.name = name;
            this.score = score;
        }
    }
}
