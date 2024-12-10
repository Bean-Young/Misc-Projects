import java.util.LinkedList;
import java.util.Scanner;

public class Main {
    public Main() {
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        LinkedList<Window> windows = new LinkedList();

        int numClicks;
        int i;
        int x;
        int y;
        int j;
        for(numClicks = 0; numClicks < n; ++numClicks) {
            i = scanner.nextInt();
            x = scanner.nextInt();
            y = scanner.nextInt();
            j = scanner.nextInt();
            int y2 = scanner.nextInt();
            windows.add(new Window(i, x, y, j, y2));
        }

        numClicks = scanner.nextInt();

        for(i = 0; i < numClicks; ++i) {
            x = scanner.nextInt();
            y = scanner.nextInt();

            for(j = 0; j <= windows.size() - 1; ++j) {
                Window window = (Window)windows.get(j);
                if (window.isInside(x, y)) {
                    windows.remove(j);
                    windows.addFirst(window);
                    break;
                }
            }
        }

        for(i = 0; i <= windows.size() - 1; ++i) {
            if (i != 0) {
                System.out.print(" ");
            }

            System.out.print(((Window)windows.get(i)).id);
        }

        scanner.close();
    }
}