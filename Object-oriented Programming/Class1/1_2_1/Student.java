public class Student extends Person {
    private double grade;

    public Student(String name, int age, double grade) {
        super(name, age);
        this.grade = grade;
    }

    public void display() {
        System.out.println("grade:" + this.grade);
    }

    public static void main(String[] args) {
        Student student = new Student("John", 20, 86.0);
        student.display();
    }
}
