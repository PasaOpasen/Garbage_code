library(tidyverse)
library(magrittr)
library(car)


df=DNase

# функция для оценки качества модели
mysummary = function(mdl) {
  cat("-----> ОБЩАЯ ИНФОРМАЦИЯ О МОДЕЛИ:\n")
  
  cat("-----> Ошибка = " ,mdl$residuals^2 %>% mean(),'\n')
  
  cat("\n")
  gvlma::gvlma(mdl) %>% summary()
  cat("\n")
  
  
  cat("-----> БАЗОВЫЕ ГРАФИКИ:\n")
  cat("\n")
  par(mfrow = c(2, 2))
  plot(mdl)
  par(mfrow = c(1, 1))
  cat("\n")
  
  
  cat("-----> ТЕСТ НА НОРМАЛЬНОСТЬ РАСПРЕДЕЛЕНИЯ ОСТАТКОВ\n")
  cat("\n")
  shapiro.test(mdl$residuals) %>% print()
  cat("\n")
  
  qqPlot(mdl, main = "Q-Q plot")
  
}

# основные статистики
psych::describe(df)  
summary(df)

# другие статистики можно посчитать примерно так
sapply(df[,-1] , function(x) sd(x))
sapply(df[,-1] , function(x) mad(x))
sapply(df[,-1] , function(x) mean(sin(x)))

# -1 -- это убрать первый столбец, который факторный, а не численный

#графики
GGally::ggpairs(df,
                lower = list(continuous = "points", combo = "box_no_facet", discrete = "facetbar", na =
                                  "na"))


ps=ggplot(df,aes(y=density,x=conc,col=Run))+
  geom_point(size=3)+
  theme_bw()

ps

ps+facet_wrap(.~Run)# из этого графика видно, что зависимость между кол. переменными для каждого уровня Run очень схожая и похожа на sqrt


summary(aov(density ~ conc+Run:conc +Run,df))# дисперсионный анализ подтверждает наличие зависимости, однако от фактора зависимости не выявлено

cor.test(df$conc,df$density) # тест на корреляцию подтверждает её наличие на весьма высоком уровне

# модели

fit1=lm(density~conc,df)

mysummary(fit1)

# эта модель статистически значима + R^2 > 80%
# но не прошла тесты на нормальность остатков,
# эксцесс и другие
#

fit2=lm(density~sqrt(conc),df)

mysummary(fit2)

# эта модель статистически значима + R^2 > 95%
# но не прошла тесты на нормальность остатков,
# надо её ещё улучшить
#

fit3=lm(density~sqrt(conc):Run,df)

mysummary(fit3)

# эта модель статистически значима + R^2 > 98%,
# взаимодействие между sqrt(conc) и всеми уровнями Run значимо
# но не прошла тесты на нормальность остатков,
# надо её ещё улучшить
#

fit4=lm(density~sqrt(conc):Run+conc:Run,df)

mysummary(fit4)

# модель ещё лучше


fit4=lm(density~sqrt(conc):Run+conc:Run+I(conc^2):Run,df)

mysummary(fit4)

# модель ещё лучше, лучше этой линейную модель уже вряд ли сделаешь



# в документации есть ещё такие нелинейные модели, но у них ошибка больше
fm1 <- nls(density ~ SSlogis( log(conc), Asym, xmid, scal ),
           data = DNase, subset = Run == 1)
## compare with a four-parameter logistic
fm2 <- nls(density ~ SSfpl( log(conc), A, B, xmid, scal ),
           data = DNase, subset = Run == 1)
summary(fm2)
anova(fm1, fm2)

(predict(fm1,df)-df$density)^2 %>% mean()
(predict(fm2,df)-df$density)^2 %>% mean()


#corrplot::corrplot(cor(df[,-1]),method='number')

#cor.test(df$conc,df$density)

#summary(aov(conc ~ .,df))

#psych::principal(df, nfactors = 10, rotate = "none")




