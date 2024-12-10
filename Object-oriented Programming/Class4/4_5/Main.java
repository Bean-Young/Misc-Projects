public class Main {
    public Main() {
    }

    public static void main(String[] args) {
        PhoneInterface phone = new PhoneInterface() {
            public void sendMsg(String msg, String number) {
                System.out.println("send{" + msg + "}to{" + number + "}");
            }

            public void callNumber(String number) {
                System.out.println("call-" + number);
            }
        };
        phone.sendMsg("你在忙吗？", "13800000000");
        phone.callNumber("13800000000");
    }
}
