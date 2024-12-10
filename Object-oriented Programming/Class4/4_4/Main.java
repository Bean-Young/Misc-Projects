import java.util.ArrayList;
import java.util.Iterator;

public class Main {
    public Main() {
    }

    public static void main(String[] args) {
        ArrayDemo2 arrayDemo = new ArrayDemo2();
        ArrayList<String> list = arrayDemo.getList();

        for(int i = 0; i < list.size(); ++i) {
            System.out.print((String)list.get(i));
            if (i < list.size() - 1) {
                System.out.print(" ");
            }
        }

        System.out.println();
        Iterator var5 = list.iterator();

        while(var5.hasNext()) {
            String element = (String)var5.next();
            System.out.print(element);
            if (!element.equals(list.get(list.size() - 1))) {
                System.out.print(" ");
            }
        }

    }
}