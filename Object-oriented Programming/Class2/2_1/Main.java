public class Main {
    private String name;
    private int id;
    private String className;
    static int count = 0;

    public Main(String name, int id, String className) {
        this.name = name;
        this.id = id;
        this.className = className;
        ++count;
    }

    public void printInfo() {
        System.out.print(this.name + ", " + this.id + ", " + this.className + "; ");
    }

    public static void main(String[] args) {
        Main student1 = new Main("s1", 17101, "171");
        Main student2 = new Main("s2", 17102, "171");
        student1.printInfo();
        student2.printInfo();
        System.out.println("count=" + count);
    }
}