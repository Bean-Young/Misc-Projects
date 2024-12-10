public class Main {
    public Main() {
    }

    public static void main(String[] args) {
        String str = "abc";
        Integer num = 0;

        try {
            num = Integer.parseInt(str);
        } catch (NumberFormatException var7) {
            num = -1;
        } finally {
            str = null;
        }

        System.out.println(str + " " + num);
    }
}
