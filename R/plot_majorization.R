library(ggplot2)

x <- seq(0,10,by=0.001)
y <- log(1+exp(-x))

#Change this gamma to change the intersection point with the majorization function
gamma <- 3
b <- (1/8)*gamma-(1/2)*(-1 / (1 + exp(gamma)))
c <- log(1+exp(-gamma)) -(1/8)*gamma^2 +2*b*gamma
ymajor <- (1/8)*x^2-2*b*x+c

temp1 <- cbind(x,y,"f(x)")
temp2 <- cbind(x,ymajor, "Majorized")
data <- as.data.frame(rbind(temp1, temp2))
colnames(data) <- c("x", "y", "Function")
data[,1] <- as.numeric(as.character(data[,1]))
data[,2] <- as.numeric(as.character(data[,2]))
data[,3] <- as.factor(data[,3])

ggplot(data, aes(x = x, y = y, color = Function))+ geom_line()+ ylim(0,1) + scale_color_discrete()

                         
                         