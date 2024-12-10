public class Main {
    public Main() {
    }

    public static void main(String[] args) {
        TicketSeller seller = new TicketSeller();
        Thread window1 = new Thread(seller, "线程0");
        Thread window2 = new Thread(seller, "线程1");
        window1.start();
        window2.start();

        try {
            window1.join();
            window2.join();
        } catch (InterruptedException var5) {
            InterruptedException e = var5;
            e.printStackTrace();
        }

    }
}