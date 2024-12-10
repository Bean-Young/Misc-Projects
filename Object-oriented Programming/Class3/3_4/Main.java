import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class Main {
    public Main() {
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String key = scanner.nextLine();
        int index = 0;
        BufferedReader br = null;
        BufferedWriter bw = null;

        try {
            br = new BufferedReader(new FileReader("E:/Java/3_4/src/1.txt"));
            bw = new BufferedWriter(new FileWriter("E:/Java/3_4/src/2.txt"));

            String line;
            while((line = br.readLine()) != null) {
                for(int i = 0; i < line.length(); ++i) {
                    bw.write(line.charAt(i) ^ key.charAt(index));
                    ++index;
                    index %= key.length();
                }

                bw.newLine();
            }
        } catch (Exception var16) {
            Exception e = var16;
            e.printStackTrace();
        } finally {
            try {
                if (br != null) {
                    br.close();
                }

                if (bw != null) {
                    bw.close();
                }

                scanner.close();
            } catch (IOException var15) {
                IOException e = var15;
                e.printStackTrace();
            }

        }

    }
}
