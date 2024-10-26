import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Stack;
import java.util.EmptyStackException;
import java.text.DecimalFormat;

public class Calculator extends JFrame implements ActionListener {
    private JButton[] buttons;
    private StringBuilder expressionBuilder; // 用于保存用户输入的表达式
    private String history; // 保存历史记录
    private boolean showResult; // 是否显示结果
    private double previousResult; // 上一次的计算结果
    private JTextField expressionField; // 用于显示表达式的文本框
    private JTextField resultField; // 用于显示结果的文本框
    private DecimalFormat df; // 格式化数字

    public Calculator() {
        setTitle("Simple Calculator");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());

        JPanel panel = new JPanel();
        panel.setLayout(new GridLayout(5, 4));

        String[] buttonLabels = {"Acc", "7", "8", "9", "/", "4", "5", "6", "*", "1", "2", "3", "-", "0", "C", "CE", "+", "√", "^2", "="};
        buttons = new JButton[buttonLabels.length];

        for (int i = 0; i < buttonLabels.length; i++) {
            buttons[i] = new JButton(buttonLabels[i]);
            buttons[i].addActionListener(this);
            panel.add(buttons[i]);
        }
        add(panel, BorderLayout.CENTER);

        // 初始化表达式和结果文本框
        expressionField = new JTextField(20);
        expressionField.setPreferredSize(new Dimension(20, 40)); // 设置更高的文本框
        expressionField.setFont(new Font(expressionField.getFont().getName(), Font.PLAIN, 24)); // 设置字体大小
        resultField = new JTextField(20);
        resultField.setEditable(false); // 结果文本框只读
        resultField.setForeground(Color.RED); // 设置文本颜色为红色
        resultField.setFont(new Font(resultField.getFont().getName(), Font.PLAIN, 24)); // 设置字体大小
        resultField.setHorizontalAlignment(SwingConstants.RIGHT); // 设置水平对齐方式为右对齐

        // 将文本框添加到界面上
        add(expressionField, BorderLayout.NORTH);
        add(resultField, BorderLayout.SOUTH);

        setVisible(true);
        pack(); // Pack the frame to fit the components

        // Initialize flags and variables
        showResult = false;
        expressionBuilder = new StringBuilder();
        history = "0"; // 默认历史记录为0
        previousResult = 0; // 默认上一次的计算结果为0

        // 初始化 DecimalFormat
        df = new DecimalFormat("#.####");
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        JButton source = (JButton) e.getSource();
        String buttonText = source.getText();

        if (buttonText.equals("C")) {
            if (expressionBuilder.length() > 0) {
                expressionBuilder.deleteCharAt(expressionBuilder.length() - 1);
            }
        } else if (buttonText.equals("CE")) {
            expressionBuilder.setLength(0);
        } else if (buttonText.equals("Acc")) {
            // 将上一次的计算结果或者默认结果输入到表达式中
            expressionBuilder.append(history);
        } else if (buttonText.matches("[0-9]+(\\.[0-9]+)?")) { // 匹配包括小数的数字
            if (showResult) {
                //expressionBuilder.setLength(0); // 清空表达式
                showResult = false;
            }
            expressionBuilder.append(buttonText);
        } else if (buttonText.equals("+") || buttonText.equals("-") || buttonText.equals("*") || buttonText.equals("/")) {
            expressionBuilder.append(buttonText);
        } else if (buttonText.equals("√")) {
            // 检查根号后面是否是负数
            if (expressionBuilder.length() == 0 || isOperator(expressionBuilder.charAt(expressionBuilder.length() - 1))) {
                JOptionPane.showMessageDialog(this, "Error: Invalid expression");
                return;
            } else {
                int lastIndex = expressionBuilder.length() - 1;
                while (lastIndex >= 0 && Character.isDigit(expressionBuilder.charAt(lastIndex))) {
                    lastIndex--;
                }
                if (lastIndex >= 0 && expressionBuilder.charAt(lastIndex) == '-') {
                    JOptionPane.showMessageDialog(this, "Error: Square root of a negative number");
                    return;
                }
            }
            expressionBuilder.append("√");
        } else if (buttonText.equals("^2")) {
            expressionBuilder.append("^2");
        } else if (buttonText.equals("=")) {
            // 当按下等号时进行计算
            String expression = expressionBuilder.toString();
            try {
                // 计算结果
                double result = evaluateExpression(expression);
                previousResult = result; // 更新上一次的计算结果
                history = df.format(previousResult); // 更新历史记录
                expressionBuilder.setLength(0); // 清空表达式
                expressionBuilder.append(history); // 将计算结果作为下一次的输入
                showResult = true; // 显示计算结果

                // 在此处执行一次"CE"操作
                expressionBuilder.setLength(0);
            } catch (ArithmeticException ex) {
                JOptionPane.showMessageDialog(this, "Error: Division by zero");
            } catch (NumberFormatException ex) {
                JOptionPane.showMessageDialog(this, "Error: Invalid expression");
            }
        }

        // 更新显示
        updateDisplay();
    }

    private boolean isOperator(char c) {
        return c == '+' || c == '-' || c == '*' || c == '/';
    }

    private void updateDisplay() {
        // 更新表达式文本框内容
        expressionField.setText(expressionBuilder.toString());
        // 更新结果文本框内容
        resultField.setText(showResult ? df.format(previousResult) : "");
    }

    private double evaluateExpression(String expression) {
        // 将表达式转换为后缀表达式
        String[] tokens = expression.split("(?<=[-+*/()^√])|(?=[-+*/()^√])");
        Stack<String> operatorStack = new Stack<>();
        StringBuilder postfix = new StringBuilder();

        for (String token : tokens) {
            if (token.matches("[0-9]+(\\.[0-9]+)?")) { // 匹配包括小数的数字
                postfix.append(token).append(" ");
            } else if (token.equals("(")) {
                operatorStack.push(token);
            } else if (token.equals(")")) {
                while (!operatorStack.peek().equals("(")) {
                    postfix.append(operatorStack.pop()).append(" ");
                }
                operatorStack.pop(); // 弹出左括号
            } else {
                // 处理运算符的优先级
                while (!operatorStack.isEmpty() && precedence(operatorStack.peek()) >= precedence(token)) {
                    postfix.append(operatorStack.pop()).append(" ");
                }
                operatorStack.push(token);
            }
        }

        while (!operatorStack.isEmpty()) {
            postfix.append(operatorStack.pop()).append(" ");
        }

        // 计算后缀表达式的值
        Stack<Double> operandStack = new Stack<>();
        String[] postfixTokens = postfix.toString().split(" ");
        for (String token : postfixTokens) {
            if (token.matches("[0-9]+(\\.[0-9]+)?")) { // 匹配包括小数的数字
                operandStack.push(Double.parseDouble(token));
            } else {
                if (token.equals("+")) {
                    double operand2 = operandStack.pop();
                    double operand1 = operandStack.pop();
                    operandStack.push(operand1 + operand2);
                } else if (token.equals("-")) {
                    double operand2 = operandStack.pop();
                    double operand1 = operandStack.pop();
                    operandStack.push(operand1 - operand2);
                } else if (token.equals("*")) {
                    double operand2 = operandStack.pop();
                    double operand1 = operandStack.pop();
                    operandStack.push(operand1 * operand2);
                } else if (token.equals("/")) {
                    double operand2 = operandStack.pop();
                    double operand1 = operandStack.pop();
                    if (operand2 == 0) {
                        throw new ArithmeticException("Division by zero");
                    }
                    operandStack.push(operand1 / operand2);
                } else if (token.equals("^")) {
                    double operand2 = operandStack.pop();
                    double operand1 = operandStack.pop();
                    operandStack.push(Math.pow(operand1, operand2));
                } else if (token.equals("√")) {
                    double operand = operandStack.pop();
                    if (operand < 0) {
                        throw new ArithmeticException("Square root of a negative number");
                    }
                    operandStack.push(Math.sqrt(operand));
                }
            }
        }

        return operandStack.pop();
    }

    private int precedence(String operator) {
        switch (operator) {
            case "+":
            case "-":
                return 1;
            case "*":
            case "/":
                return 2;
            case "^":
            case "√":
                return 3;
            default:
                return 0;
        }
    }

    public static void main(String[] args) {
        new Calculator();
    }
}


