import java.io.PrintStream;

class TicketSeller implements Runnable {
    private static int tickets = 10;
    private static final Object lock = new Object();
    private static boolean turn = true;

    TicketSeller() {
    }

    public void run() {
        while(true) {
            if (tickets > 0) {
                synchronized(lock) {
                    while(Thread.currentThread().getName().equals("线程0") && !turn || Thread.currentThread().getName().equals("线程1") && turn) {
                        try {
                            lock.wait();
                        } catch (InterruptedException var4) {
                            return;
                        }
                    }

                    if (tickets > 0) {
                        PrintStream var10000 = System.out;
                        String var10001 = Thread.currentThread().getName();
                        var10000.println(var10001 + "卖出一张票，余票" + --tickets);
                        turn = !turn;
                        lock.notifyAll();
                        continue;
                    }

                    lock.notifyAll();
                    return;
                }
            }

            return;
        }
    }
}