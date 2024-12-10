import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.text.DecimalFormat;
import java.util.Stack;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;

public class Calculator extends JFrame implements ActionListener {
    private JButton[] buttons;
    private StringBuilder expressionBuilder;
    private String history;
    private boolean showResult;
    private double previousResult;
    private JTextField expressionField;
    private JTextField resultField;
    private DecimalFormat df;

    public Calculator() {
        this.setTitle("Simple Calculator");
        this.setDefaultCloseOperation(3);
        this.setLayout(new BorderLayout());
        JPanel panel = new JPanel();
        panel.setLayout(new GridLayout(5, 4));
        String[] buttonLabels = new String[]{"Acc", "7", "8", "9", "/", "4", "5", "6", "*", "1", "2", "3", "-", "0", "C", "CE", "+", "√", "^2", "="};
        this.buttons = new JButton[buttonLabels.length];

        for(int i = 0; i < buttonLabels.length; ++i) {
            this.buttons[i] = new JButton(buttonLabels[i]);
            this.buttons[i].addActionListener(this);
            panel.add(this.buttons[i]);
        }

        this.add(panel, "Center");
        this.expressionField = new JTextField(20);
        this.expressionField.setPreferredSize(new Dimension(20, 40));
        this.expressionField.setFont(new Font(this.expressionField.getFont().getName(), 0, 24));
        this.resultField = new JTextField(20);
        this.resultField.setEditable(false);
        this.resultField.setForeground(Color.RED);
        this.resultField.setFont(new Font(this.resultField.getFont().getName(), 0, 24));
        this.resultField.setHorizontalAlignment(4);
        this.add(this.expressionField, "North");
        this.add(this.resultField, "South");
        this.setVisible(true);
        this.pack();
        this.showResult = false;
        this.expressionBuilder = new StringBuilder();
        this.history = "0";
        this.previousResult = 0.0;
        this.df = new DecimalFormat("#.####");
    }

    public void actionPerformed(ActionEvent e) {
        JButton source = (JButton)e.getSource();
        String buttonText = source.getText();
        if (buttonText.equals("C")) {
            if (this.expressionBuilder.length() > 0) {
                this.expressionBuilder.deleteCharAt(this.expressionBuilder.length() - 1);
            }
        } else if (buttonText.equals("CE")) {
            this.expressionBuilder.setLength(0);
        } else if (buttonText.equals("Acc")) {
            this.expressionBuilder.append(this.history);
        } else if (buttonText.matches("[0-9]+(\\.[0-9]+)?")) {
            if (this.showResult) {
                this.showResult = false;
            }

            this.expressionBuilder.append(buttonText);
        } else if (!buttonText.equals("+") && !buttonText.equals("-") && !buttonText.equals("*") && !buttonText.equals("/")) {
            if (buttonText.equals("√")) {
                if (this.expressionBuilder.length() == 0 || this.isOperator(this.expressionBuilder.charAt(this.expressionBuilder.length() - 1))) {
                    JOptionPane.showMessageDialog(this, "Error: Invalid expression");
                    return;
                }

                int lastIndex;
                for(lastIndex = this.expressionBuilder.length() - 1; lastIndex >= 0 && Character.isDigit(this.expressionBuilder.charAt(lastIndex)); --lastIndex) {
                }

                if (lastIndex >= 0 && this.expressionBuilder.charAt(lastIndex) == '-') {
                    JOptionPane.showMessageDialog(this, "Error: Square root of a negative number");
                    return;
                }

                this.expressionBuilder.append("√");
            } else if (buttonText.equals("^2")) {
                this.expressionBuilder.append("^2");
            } else if (buttonText.equals("=")) {
                String expression = this.expressionBuilder.toString();

                try {
                    double result = this.evaluateExpression(expression);
                    this.previousResult = result;
                    this.history = this.df.format(this.previousResult);
                    this.expressionBuilder.setLength(0);
                    this.expressionBuilder.append(this.history);
                    this.showResult = true;
                    this.expressionBuilder.setLength(0);
                } catch (ArithmeticException var7) {
                    JOptionPane.showMessageDialog(this, "Error: Division by zero");
                } catch (NumberFormatException var8) {
                    JOptionPane.showMessageDialog(this, "Error: Invalid expression");
                }
            }
        } else {
            this.expressionBuilder.append(buttonText);
        }

        this.updateDisplay();
    }

    private boolean isOperator(char c) {
        return c == '+' || c == '-' || c == '*' || c == '/';
    }

    private void updateDisplay() {
        this.expressionField.setText(this.expressionBuilder.toString());
        this.resultField.setText(this.showResult ? this.df.format(this.previousResult) : "");
    }

    private double evaluateExpression(String expression) {
        String[] tokens = expression.split("(?<=[-+*/()^√])|(?=[-+*/()^√])");
        Stack<String> operatorStack = new Stack();
        StringBuilder postfix = new StringBuilder();
        String[] var5 = tokens;
        int var6 = tokens.length;

        for(int var7 = 0; var7 < var6; ++var7) {
            String token = var5[var7];
            if (token.matches("[0-9]+(\\.[0-9]+)?")) {
                postfix.append(token).append(" ");
            } else if (token.equals("(")) {
                operatorStack.push(token);
            } else if (token.equals(")")) {
                while(!((String)operatorStack.peek()).equals("(")) {
                    postfix.append((String)operatorStack.pop()).append(" ");
                }

                operatorStack.pop();
            } else {
                while(!operatorStack.isEmpty() && this.precedence((String)operatorStack.peek()) >= this.precedence(token)) {
                    postfix.append((String)operatorStack.pop()).append(" ");
                }

                operatorStack.push(token);
            }
        }

        while(!operatorStack.isEmpty()) {
            postfix.append((String)operatorStack.pop()).append(" ");
        }

        Stack<Double> operandStack = new Stack();
        String[] postfixTokens = postfix.toString().split(" ");
        String[] var17 = postfixTokens;
        int var18 = postfixTokens.length;

        for(int var9 = 0; var9 < var18; ++var9) {
            String token = var17[var9];
            if (token.matches("[0-9]+(\\.[0-9]+)?")) {
                operandStack.push(Double.parseDouble(token));
            } else {
                double operand;
                double operand1;
                if (token.equals("+")) {
                    operand = (Double)operandStack.pop();
                    operand1 = (Double)operandStack.pop();
                    operandStack.push(operand1 + operand);
                } else if (token.equals("-")) {
                    operand = (Double)operandStack.pop();
                    operand1 = (Double)operandStack.pop();
                    operandStack.push(operand1 - operand);
                } else if (token.equals("*")) {
                    operand = (Double)operandStack.pop();
                    operand1 = (Double)operandStack.pop();
                    operandStack.push(operand1 * operand);
                } else if (token.equals("/")) {
                    operand = (Double)operandStack.pop();
                    operand1 = (Double)operandStack.pop();
                    if (operand == 0.0) {
                        throw new ArithmeticException("Division by zero");
                    }

                    operandStack.push(operand1 / operand);
                } else if (token.equals("^")) {
                    operand = (Double)operandStack.pop();
                    operand1 = (Double)operandStack.pop();
                    operandStack.push(Math.pow(operand1, operand));
                } else if (token.equals("√")) {
                    operand = (Double)operandStack.pop();
                    if (operand < 0.0) {
                        throw new ArithmeticException("Square root of a negative number");
                    }

                    operandStack.push(Math.sqrt(operand));
                }
            }
        }

        return (Double)operandStack.pop();
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
