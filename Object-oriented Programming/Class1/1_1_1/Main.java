public class Main {
    public Main() {
    }

    public static void main(String[] args) {
        ChanghongTV myTV = new ChanghongTV(60, "BTV");
        System.out.println("Changhong, " + myTV.size + "inch");
        System.out.print("current channel is " + myTV.currentChannel + ", ");
        myTV.switchChannel("NTV");
    }
}
