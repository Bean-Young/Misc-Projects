import java.util.ArrayList;
import java.util.Iterator;

public class Test {
    public Test() {
    }

    public static void main(String[] args) {
        ArrayList<String> list = new ArrayList();
        list.add("it");
        list.add("is");
        list.add("so");
        list.add("cool");

        for(int i = 0; i < list.size(); ++i) {
            System.out.print((String)list.get(i));
            if (i < list.size() - 1) {
                System.out.print(" ");
            }
        }

        System.out.println();
        Iterator var4 = list.iterator();

        while(var4.hasNext()) {
            String element = (String)var4.next();
            System.out.print(element);
            System.out.print(" ");
        }

    }
}