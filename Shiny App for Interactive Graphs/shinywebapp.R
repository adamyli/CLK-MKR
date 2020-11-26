library(shiny)
library(shinythemes)
library(ggplot2)
options(shiny.maxRequestSize = 100000*1024^2)

ui <- fluidPage(theme = shinytheme("superhero"),
  titlePanel("CLK-MKR: A machine learning toolkit for optimal feature selection and epigenetic clock building "),
  sidebarLayout(
    sidebarPanel(
      
      fileInput("file1", "Upload Methylation Data CSV File"),
      actionButton('run', 'Run'),
      actionButton('plotinput', 'Visualize Input Data'),
      selectInput('x_axis', label = 'X Axis', choices = ('No choices here yet')),
      selectInput('y_axis', label = 'Y Axis', choices = ('No choices here yet')),
      
      tags$hr(),
      
      checkboxInput("header", "Uploaded Methylation Data Header & Viewing Control", TRUE),
      radioButtons("disp", "Display",
                   choices = c(Head = "head",
                               All = "all"),
                   selected = "head"),
      tags$hr(),
      
      downloadButton("downloadModel", "Download Best Model (PKL)"),
      tags$hr(),
      downloadButton("downloadParameters", "Download Best Model Parameters (CSV)"),
      tags$hr(),
      downloadButton("downloadCpGs", "Download Selected CpGs (CSV)"),
      #width = 5
      tags$hr(),
      tags$hr(),
      actionButton('plotresult', 'Visualize Results'),
      tags$hr(),
      checkboxInput("cpg_header", "Model Scores and Selected CpGs Header & Viewing Control", TRUE),
      radioButtons("cpg_buttons", "Display",
                   choices = c(Head = "head",
                               All = "all"),
                   selected = "head"),
      selectInput('labelled_x_axis', label = 'X Axis', choices = ('No choices here yet')),
      selectInput('labelled_y_axis', label = 'Y Axis', choices = ('No choices here yet')),
    ),
    mainPanel(
      # Output: Data file ----
      tableOutput("contents"),
      tags$hr(),
      plotOutput("input_data"),
      tags$hr(),
      tableOutput("cpgs"),
      tags$hr(),
      plotOutput("labelled_graph"),
      tags$hr(),
      plotOutput("age_graph")
    )
  )
)

# Define server logic to read selected file ----
server <- function(input, output, session) {

  observeEvent(input$plotinput, {
    req(input$file1)
    
    #Render interactive plot of uploaded CSV file
    mytable <- read.csv(input$file1$datapath)
    
    updateSelectInput(session, "x_axis", label = 'X Axis', choices = colnames(mytable))
    updateSelectInput(session, "y_axis", label = 'Y Axis', choices = colnames(mytable))
    
    output$input_data <- renderPlot({
      req(input$file1)
      methy_data <- read.csv(input$file1$datapath)
      ggplot(data=methy_data, aes_string(x=input$x_axis,y=input$y_axis)) + geom_point(size = 2.5, colour='#E37710')+
        theme(text = element_text(size=15))
    })
  })
  
  observeEvent(input$run, {
    req(input$file1)

    #Run the .py script and create graphs out of the results
    system(paste("python webapp.py ",input$file1$datapath," > data_out.txt"))
  })
  
  observeEvent(input$plotresult, {
    req(input$file1)
    labelled_table <- read.csv('labelled_best_cpgs.csv')
    
    updateSelectInput(session, 'labelled_x_axis', label = 'X Axis', choices = colnames(labelled_table))
    updateSelectInput(session, 'labelled_y_axis', label = 'Y Axis', choices = colnames(labelled_table))

  output$labelled_graph <- renderPlot({
    age_table <- read.csv('labelled_best_cpgs.csv')
    ggplot(data=age_table, aes_string(x=input$labelled_x_axis,y=input$labelled_y_axis))+ geom_point(size = 3, aes(color = Age.Labels)) +
      geom_smooth(aes(color = Age.Labels, fill = Age.Labels), method = "lm") + theme(text = element_text(size=15)) 
  })
  
  output$age_graph <- renderPlot({
    age_table <- read.csv('age_graph.csv')
    ggplot(data=age_table, aes_string(x='Chronological.Age',y='Predicted.Age')) + geom_point(size = 3, aes(color = Age.Labels)) + 
      geom_smooth(aes(color = Age.Labels, fill = Age.Labels), method = "lm") +theme(text = element_text(size=15)) 
  })
  output$cpgs <- renderTable({  df <- read.csv('final_cpg_list.csv',
                                               header = input$cpg_header,
                                               sep = ',')
  if(input$cpg_buttons == "head") {return(head(df))}
  else {return(df)}})

  
  #The 3 downloadable files
  output$downloadModel <- downloadHandler(
    filename <- function() {
      paste("final_clock_model", "pkl", sep=".")
    },
    content <- function(file) {
      file.copy("final_clock_model.pkl", file)
    }
  )
  
  output$downloadParameters <- downloadHandler(
    filename <- function() {
      paste("best_model_parameters", "csv", sep=".")
    },
    content <- function(file) {
      file.copy("best_model_parameters.csv", file)
    }
  )
  
  output$downloadCpGs <- downloadHandler(
    filename <- function() {
      paste("final_cpg_list", "csv", sep=".")
    },
    content <- function(file) {
      file.copy("final_cpg_list.csv", file)
    }
  )
    
  })

}

# Create Shiny app ----
shinyApp(ui, server)