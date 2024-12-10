import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

public class Main {
    public Main() {
    }

    public static void main(String[] args) {
        Set<String> dictionary = new HashSet();
        List<String> errors = new ArrayList();

        IOException e;
        String line;
        BufferedReader br;
        try {
            br = new BufferedReader(new FileReader("E:/Java/3_3/src/index.txt"));

            try {
                while((line = br.readLine()) != null) {
                    dictionary.add(line.toLowerCase());
                }
            } catch (Throwable var18) {
                try {
                    br.close();
                } catch (Throwable var11) {
                    var18.addSuppressed(var11);
                }

                throw var18;
            }

            br.close();
        } catch (IOException var19) {
            e = var19;
            e.printStackTrace();
            return;
        }

        try {
            br = new BufferedReader(new FileReader("E:/Java/3_3/src/in.txt"));

            try {
                while((line = br.readLine()) != null) {
                    String[] words = line.replaceAll("\\d", "").split("\\W+");
                    String[] var6 = words;
                    int var7 = words.length;

                    for(int var8 = 0; var8 < var7; ++var8) {
                        String word = var6[var8];
                        String lowerCaseWord = word.toLowerCase();
                        if (!lowerCaseWord.isEmpty() && !dictionary.contains(lowerCaseWord)) {
                            errors.add(lowerCaseWord);
                        }
                    }
                }
            } catch (Throwable var16) {
                try {
                    br.close();
                } catch (Throwable var12) {
                    var16.addSuppressed(var12);
                }

                throw var16;
            }

            br.close();
        } catch (IOException var17) {
            e = var17;
            e.printStackTrace();
            return;
        }

        Collections.sort(errors);

        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter("E:/Java/3_3/src/error.txt"));

            try {
                Iterator var22 = errors.iterator();

                while(var22.hasNext()) {
                    String error = (String)var22.next();
                    bw.write(error);
                    bw.newLine();
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
        } catch (IOException var15) {
            e = var15;
            e.printStackTrace();
        }

    }
}
