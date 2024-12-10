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
        int colonPosition = scanner.nextInt();
        scanner.close();

        try {
            BufferedReader br = new BufferedReader(new FileReader("E:/Java/4_3/src/listin.txt"));

            try {
                BufferedWriter bw = new BufferedWriter(new FileWriter("E:/Java/4_3/src/listout.txt"));

                String line;
                try {
                    while((line = br.readLine()) != null) {
                        String[] parts = line.split(":", 2);
                        if (parts.length == 2) {
                            String leftPart = parts[0].trim().replaceAll("\\s+", " ");
                            String rightPart = parts[1].trim().replaceAll("\\s+", " ");
                            int spaceCount = colonPosition - leftPart.length() - 1;
                            StringBuilder sb = new StringBuilder();

                            for(int i = 0; i < spaceCount; ++i) {
                                sb.append(" ");
                            }

                            sb.append(leftPart).append(" : ").append(rightPart);
                            bw.write(sb.toString());
                            bw.newLine();
                        }
                    }
                } catch (Throwable var14) {
                    try {
                        bw.close();
                    } catch (Throwable var13) {
                        var14.addSuppressed(var13);
                    }

                    throw var14;
                }

                bw.close();
            } catch (Throwable var15) {
                try {
                    br.close();
                } catch (Throwable var12) {
                    var15.addSuppressed(var12);
                }

                throw var15;
            }

            br.close();
        } catch (IOException var16) {
            IOException e = var16;
            e.printStackTrace();
        }

    }
}