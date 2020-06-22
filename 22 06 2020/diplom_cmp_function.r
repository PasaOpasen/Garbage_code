# 
#
#         Example of comparing 3 groups
#
#
install.packages("binom")
# ---------------------- set environment --------------------------------- 
WD           = "C:/work"
PATH         = WD
project.dir  = "/diplom/"

path.dir.in       = paste0(WD, project.dir, "data_in/")
path.dir.out      = paste0(WD, project.dir, "data_res/")
path.pics.out     = paste0(WD, project.dir, "pics_res/")


# include funcs
path         = paste0(WD, project.dir, 'cmp_function2.R')
source(path)
# ---------------------- set environment --------------------------------- 

?datasets::mtcars

cmp.list =  list( list(id = 1, type = "num"), #mpg	 Miles/(US) gallon
                  list(id = 2, type = "cat"), #cyl	 Number of cylinders
                  list(id = 3, type = "num"), #disp	 Displacement (cu.in.)
                  list(id = 4, type = "num"), #hp	 Gross horsepower
                  list(id = 5, type = "num"), #drat	 Rear axle ratio
                  list(id = 6, type = "num"), #wt	 Weight (1000 lbs)  
                  list(id = 7, type = "num"), #qsec	 1/4 mile time  
                  list(id = 8, type = "bin"), #vs	 Engine (0 = V-shaped, 1 = straight)
                  list(id = 9, type = "bin"), #Transmission (0 = automatic, 1 = manual)
                  list(id = 11, type = "cat") #carb	 Number of carburetors
                  )


cmp.2group.all.vars(     id.group     = 10, #gear	 Number of forward gears
                         group.labels = c("3","4","5"),
                         columns.list = cmp.list,
                         data         = datasets::mtcars, 
                         PATH         = path.dir.out, 
                         FILENAME     = "mtcars_by_gears", 
                         result       = NULL
)

