import java.util.Scanner;

public class Main {
    public Main() {
    }

    public static int f(int x, int y, int z) {
        return x >= 0 && y >= 0 && z >= 0 ? x * x + y * y + z * z : f(x, y);
    }

    public static int f(int x, int y) {
        return x >= 0 && y >= 0 ? x * x + y * y : f(x);
    }

    public static int f(int x) {
        return x >= 0 ? x * x : 0;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int x = scanner.nextInt();
        int y = scanner.nextInt();
        int z = scanner.nextInt();
        scanner.close();
        System.out.println(f(x, y, z));
    }
}
