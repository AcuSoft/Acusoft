var datos_objeto = [];
var lista_objetos = [];
var n = 0;
var myVar;

function AddPrompt(){
	this.render = function(){	               
                var winW = window.innerWidth;
                var winH = window.innerHeight;	
                
                var dialogoverlay = document.getElementById('dialogoverlay');
                var dialogbox = document.getElementById('dialogbox');
                
                dialogoverlay.style.display = "block";
                dialogoverlay.style.height = winH+"px";
                dialogbox.style.left = (winW/2)-(560*.5)+"px";
                dialogbox.style.top = "5px";
                dialogbox.style.display = "block";
        }
        
        const anadir = document.getElementById("anadir");
        const cancelar = document.getElementById("cancelar")

        anadir.addEventListener('click',()=>{
                add_prompt.ok();       
        })

        cancelar.addEventListener('click',()=>{
                add_prompt.cancel();
        })

	this.cancel = function(){
		document.getElementById('dialogoverlay').style.display = "none";
		document.getElementById('dialogbox').style.display = "none";
	}
	this.ok = function(){
                n++;
                var nombre_objeto =     document.getElementById('nombre_objeto').value;
                var material_objeto =   document.getElementById('material_objeto').value;
                var ancho_objeto =      document.getElementById('ancho_objeto').value;
                var largo_objeto =      document.getElementById('largo_objeto').value;
                var area_objeto =       document.getElementById('ancho_objeto').value*document.getElementById('largo_objeto').value;
                
                if ($("#" + Tabla + " tbody").length == 0) {
                        $("#" + Tabla).append("<tbody></tbody>");
                    }
                
                    // Append product to the table
                    $("#" + Tabla + " tbody").append(
                        "<tr>" +
                        "<td id = 'td_nombreobjeto"+ n + "'>" + nombre_objeto + 
                        "<button id = 'delete_objeto' class = 'boton2'"+
                        "type='button' onclick='productDelete(this);'>"+
                        "</td>" +
                        "<td id = 'td_materialobjeto"+ n + "'>" + material_objeto + "</td>" +
                        "<td id = 'td_areaobjeto"+ n + "'>" + area_objeto + " m<sup>2</sup></p></td>" +
                        "<td id = 'td_sa1objeto"+ n + "'></td>" +
                        "<td id = 'td_sa2objeto"+ n + "'></td>" +
                        "<td id = 'td_sa3objeto"+ n + "'></td>" +
                        "<td id = 'td_sa4objeto"+ n + "'></td>" +
                        "<td id = 'td_sa5objeto"+ n + "'></td>" +
                        "<td id = 'td_sa6objeto"+ n + "'></td>" +
                        "</tr>");

                datos_objeto.push(n,nombre_objeto,Pared,ancho_objeto,largo_objeto,material_objeto);
                
                lista_objetos.push(datos_objeto);
                datos_objeto = [];
                
                myVar = JSON.stringify(lista_objetos);           
		
		document.getElementById('dialogoverlay').style.display = "none";
                document.getElementById('dialogbox').style.display = "none";
                
        }
        
}
var add_prompt = new AddPrompt();

add_objeto.addEventListener('click', () => {
        add_prompt.render();
        $("#nombre_objeto").val("");
        $("#largo_objeto").val("");
        $("#ancho_objeto").val("");
})



$(document).ready(function(){
        $("#RT60_Form").submit(function(e){
                e.preventDefault();
                $.ajax({
                        url: $(this).attr('action'),
                        type: $(this).attr('method'),
                        data: {
                                'x': $('#x').val(),
                                'y': $('#y').val(),
                                'z': $('#z').val(),
                                'Material_T': $('#Material_T option:selected').val(),
                                'Material_P': $('#Material_P option:selected').val(),
                                'Material_WF': $('#Material_WF option:selected').val(),
                                'Material_WB': $('#Material_WB option:selected').val(),
                                'Material_WL': $('#Material_WL option:selected').val(),
                                'Material_WR': $('#Material_WR option:selected').val(),
                                'Extras':myVar
                        },

                        success: function(datos){                              
                                
                                $("#td_sT" ).html("<p>" + datos.sT  + " m<sup>2</sup></p>");
                                $("#td_sP" ).html("<p>" + datos.sP  + " m<sup>2</sup></p>");
                                $("#td_sWF").html("<p>" + datos.sWF + " m<sup>2</sup></p>");
                                $("#td_sWB").html("<p>" + datos.sWB + " m<sup>2</sup></p>");
                                $("#td_sWL").html("<p>" + datos.sWL + " m<sup>2</sup></p>");
                                $("#td_sWR").html("<p>" + datos.sWR + " m<sup>2</sup></p>");

                                for (var i = 0; i<6;i++){
                                        $("#td_saT"+i).html("<p>" + datos.saT[i] + " </p>");
                                }
                                for (var i = 0; i<6;i++){
                                        $("#td_saP"+i).html("<p>" + datos.saP[i] + " </p>");
                                }
                                for (var i = 0; i<6;i++){
                                        $("#td_saWF"+i).html("<p>" + datos.saWF[i] + " </p>");
                                }
                                for (var i = 0; i<6;i++){
                                        $("#td_saWB"+i).html("<p>" + datos.saWB[i] + " </p>");
                                }
                                for (var i = 0; i<6;i++){
                                        $("#td_saWL"+i).html("<p>" + datos.saWL[i] + " </p>");
                                }
                                for (var i = 0; i<6;i++){
                                        $("#td_saWR"+i).html("<p>" + datos.saWR[i] + " </p>");
                                }
                                for (var i = 0; i<6;i++){
                                        $("#td_saS"+i).html("<p>" + datos.RT60f[0][i] + " s</p>");
                                }

                                for (var c = 1; c=lista_objetos.length; c++){
                                        for(var i = 0; i<6;i++){
                                                $("td_sa"+i+"objeto"+c).html("<p>" +  + " </p>");
                                        }
                                }
                                T.addEventListener('click', () => { 
                                        document.getElementById("td_Material").innerHTML = datos.matT
                                        for (var i = 1; i<7;i++){
                                                document.getElementById("td_C"+i).innerHTML = datos.cT[i-1];
                                        }                                        
                                })
                                P.addEventListener('click', () => { 
                                        document.getElementById("td_Material").innerHTML = datos.matP
                                        for (var i = 1; i<7;i++){
                                                document.getElementById("td_C"+i).innerHTML = datos.cP[i-1];
                                        }                                        
                                })
                                WF.addEventListener('click', () => { 
                                        document.getElementById("td_Material").innerHTML = datos.matWF
                                        for (var i = 1; i<7;i++){
                                                document.getElementById("td_C"+i).innerHTML = datos.cWF[i-1];
                                        }                                        
                                })
                                WB.addEventListener('click', () => { 
                                        document.getElementById("td_Material").innerHTML = datos.matWB
                                        for (var i = 1; i<7;i++){
                                                document.getElementById("td_C"+i).innerHTML = datos.cWB[i-1];
                                        }                                        
                                })
                                WL.addEventListener('click', () => { 
                                        document.getElementById("td_Material").innerHTML = datos.matWL
                                        for (var i = 1; i<7;i++){
                                                document.getElementById("td_C"+i).innerHTML = datos.cWL[i-1];
                                        }                                        
                                })
                                WR.addEventListener('click', () => { 
                                        document.getElementById("td_Material").innerHTML = datos.matWR
                                        for (var i = 1; i<7;i++){
                                                document.getElementById("td_C"+i).innerHTML = datos.cWR[i-1];
                                        }                                        
                                })

                                $("#RT60").html("<h2>" + datos.RT60 + " s</h2>");
                                $("#desv").html("<p>" + datos.dev + " s</p>");

                                var datos_RT60f = datos.RT60f[0];

                                var ctx=document.getElementById("Grafica_RT60").getContext("2d");
                                var Grafica_RT60 = new Chart(ctx,{
                                type:"line",
                                axisY:{
                                        stripLines:[
                                        {                
                                                startValue:25,
                                                endValue:32,                
                                                color:"#d8d8d8"                
                                        }
                                        ]                    
                                },
                                data:{  
                                        
                                        labels:[125,250,500,1000,2000,4000],
                                        datasets:[{
                                                label:"RT60",
                                                data:datos_RT60f,
                                                lineTension:0,
                                                backgroundColor:[
                                                        'rgb(255,192,0)',
                                                        'rgb(255,192,0)',
                                                        'rgb(255,192,0)',
                                                        'rgb(255,192,0)',
                                                        'rgb(255,192,0)',
                                                        'rgb(255,192,0)',
                                                        ],
                                                fill: false,
                                                pointRadius:5,
                                                aspectRatio:1,
                                                
                                                }]
                                        },
                              
                                options:{
                                        responsive:true,
                                        aspectRatio:2,
                                        scales:{
                                        yAxes:[{
                                                ticks:{
                                                        beginAtZero:true,
                                                        stepSize:0.20
                                                },
                                                
                                        }],
                                        

                                        
                                        }
                                },
                                
                                })
                                },
                        error: function(){
                                console.log("error");
                        }      
                })
        })
})


const x = document.getElementById("x");
const y = document.getElementById("y");
const z = document.getElementById("z");

x.addEventListener('input',()=>{
        v = x.value*y.value*z.value;
        a = 2*(x.value*y.value+x.value*z.value+y.value*z.value)
        document.getElementById("td_Volumen").innerHTML=v;
        document.getElementById("td_Area").innerHTML=a;
        document.getElementById("td_Objeto").innerHTML = "";
        document.getElementById("td_Pared").innerHTML = "";
        document.getElementById("td_Largo").innerHTML = "";
        document.getElementById("td_Ancho").innerHTML = "";
        document.getElementById("td_Superficie").innerHTML = "";  
})
y.addEventListener('input',()=>{
        v = x.value*y.value*z.value;
        a = 2*(x.value*y.value+x.value*z.value+y.value*z.value)
        document.getElementById("td_Volumen").innerHTML=v;
        document.getElementById("td_Area").innerHTML=a;
        document.getElementById("td_Objeto").innerHTML = "";
        document.getElementById("td_Pared").innerHTML = "";
        document.getElementById("td_Largo").innerHTML = "";
        document.getElementById("td_Ancho").innerHTML = "";
        document.getElementById("td_Superficie").innerHTML = ""; 
})
z.addEventListener('input',()=>{
        v = x.value*y.value*z.value;
        a = 2*(x.value*y.value+x.value*z.value+y.value*z.value)
        document.getElementById("td_Volumen").innerHTML=v;
        document.getElementById("td_Area").innerHTML=a;
        document.getElementById("td_Objeto").innerHTML = "";
        document.getElementById("td_Pared").innerHTML = "";
        document.getElementById("td_Largo").innerHTML = "";
        document.getElementById("td_Ancho").innerHTML = "";
        document.getElementById("td_Superficie").innerHTML = ""; 
})

const Material_T = document.getElementById('Material_T');
const Material_P = document.getElementById('Material_P');
const Material_WF = document.getElementById('Material_WF');
const Material_WB = document.getElementById('Material_WB');
const Material_WL = document.getElementById('Material_WL');
const Material_WR = document.getElementById('Material_WR');

Material_T.addEventListener('input',() => {
        document.getElementById("td_Objeto").innerHTML = "";
        document.getElementById("td_Pared").innerHTML = "";
        document.getElementById("td_Largo").innerHTML = "";
        document.getElementById("td_Ancho").innerHTML = "";
        document.getElementById("td_Superficie").innerHTML = ""; 
        document.getElementById("td_Material").innerHTML = ""      
})

Material_P.addEventListener('input',() => {
        document.getElementById("td_Objeto").innerHTML = "";
        document.getElementById("td_Pared").innerHTML = "";
        document.getElementById("td_Largo").innerHTML = "";
        document.getElementById("td_Ancho").innerHTML = "";
        document.getElementById("td_Superficie").innerHTML = ""; 
        document.getElementById("td_Material").innerHTML = ""      
})

Material_WF.addEventListener('input',() => {
        document.getElementById("td_Objeto").innerHTML = "";
        document.getElementById("td_Pared").innerHTML = "";
        document.getElementById("td_Largo").innerHTML = "";
        document.getElementById("td_Ancho").innerHTML = "";
        document.getElementById("td_Superficie").innerHTML = ""; 
        document.getElementById("td_Material").innerHTML = ""      
})

Material_WB.addEventListener('input',() => {
        document.getElementById("td_Objeto").innerHTML = "";
        document.getElementById("td_Pared").innerHTML = "";
        document.getElementById("td_Largo").innerHTML = "";
        document.getElementById("td_Ancho").innerHTML = "";
        document.getElementById("td_Superficie").innerHTML = ""; 
        document.getElementById("td_Material").innerHTML = ""      
})

Material_WL.addEventListener('input',() => {
        document.getElementById("td_Objeto").innerHTML = "";
        document.getElementById("td_Pared").innerHTML = "";
        document.getElementById("td_Largo").innerHTML = "";
        document.getElementById("td_Ancho").innerHTML = "";
        document.getElementById("td_Superficie").innerHTML = ""; 
        document.getElementById("td_Material").innerHTML = ""      
})

Material_WR.addEventListener('input',() => {
        document.getElementById("td_Objeto").innerHTML = "";
        document.getElementById("td_Pared").innerHTML = "";
        document.getElementById("td_Largo").innerHTML = "";
        document.getElementById("td_Ancho").innerHTML = "";
        document.getElementById("td_Superficie").innerHTML = ""; 
        document.getElementById("td_Material").innerHTML = ""      
})

const T  = document.getElementById("td_T" );
const P  = document.getElementById("td_P" );
const WF = document.getElementById("td_WF");
const WB = document.getElementById("td_WB");
const WL = document.getElementById("td_WL");
const WR = document.getElementById("td_WR");

T.addEventListener('click', () => {
        Objeto = "Pared";
        Pared  = "Techo";
        Largo  = z.value;
        Ancho  = x.value;
        Superficie = Largo*Ancho;
        Tabla = "Tabla_T";
        document.getElementById("td_Objeto").innerHTML = Objeto;
        document.getElementById("td_Pared").innerHTML = Pared;
        document.getElementById("td_Largo").innerHTML = Largo;
        document.getElementById("td_Ancho").innerHTML = Ancho;
        document.getElementById("td_Superficie").innerHTML = Superficie;
        document.getElementById("add_objeto").disabled = false;
        
})

P.addEventListener('click', () => {
        Objeto = "Pared";
        Pared  = "Piso";
        Largo  = z.value;
        Ancho  = x.value;
        Superficie = Largo*Ancho;
        Tabla = "Tabla_P";
        document.getElementById("td_Objeto").innerHTML = Objeto;
        document.getElementById("td_Pared").innerHTML = Pared;
        document.getElementById("td_Largo").innerHTML = Largo;
        document.getElementById("td_Ancho").innerHTML = Ancho;
        document.getElementById("td_Superficie").innerHTML = Superficie;
        document.getElementById("add_objeto").disabled = false;
               
})

WF.addEventListener('click', () => {
        Objeto = "Pared";
        Pared  = "Pared Frontal";
        Largo  = y.value;
        Ancho  = x.value;
        Superficie = Largo*Ancho;
        Tabla = "Tabla_WF";
        document.getElementById("td_Objeto").innerHTML = Objeto;
        document.getElementById("td_Pared").innerHTML = Pared;
        document.getElementById("td_Largo").innerHTML = Largo;
        document.getElementById("td_Ancho").innerHTML = Ancho;
        document.getElementById("td_Superficie").innerHTML = Superficie; 
        document.getElementById("td_Material").innerHTML = Material_WF.value;
        document.getElementById("add_objeto").disabled = false;
             
})

WB.addEventListener('click', () => {
        Objeto = "Pared";
        Pared  = "Pared Trasera";
        Largo  = y.value;
        Ancho  = x.value;
        Superficie = Largo*Ancho;
        Tabla = "Tabla_WB";
        document.getElementById("td_Objeto").innerHTML = Objeto;
        document.getElementById("td_Pared").innerHTML = Pared;
        document.getElementById("td_Largo").innerHTML = Largo;
        document.getElementById("td_Ancho").innerHTML = Ancho;
        document.getElementById("td_Superficie").innerHTML = Superficie;
        document.getElementById("td_Material").innerHTML = Material_WB.value;
        document.getElementById("add_objeto").disabled = false;
                 
})

WL.addEventListener('click', () => {
        Objeto = "Pared";
        Pared  = "Pared Izquierda";
        Largo  = y.value;
        Ancho  = z.value;
        Superficie = Largo*Ancho;
        Tabla = "Tabla_WL";
        document.getElementById("td_Objeto").innerHTML = Objeto;
        document.getElementById("td_Pared").innerHTML = Pared;
        document.getElementById("td_Largo").innerHTML = Largo;
        document.getElementById("td_Ancho").innerHTML = Ancho;
        document.getElementById("td_Superficie").innerHTML = Superficie;
        document.getElementById("td_Material").innerHTML = Material_WL.value;
        document.getElementById("add_objeto").disabled = false;
                
})

WR.addEventListener('click', () => {
        Objeto = "Pared";
        Pared  = "Pared Derecha";
        Largo  = y.value;
        Ancho  = z.value;
        Superficie = Largo*Ancho;
        Tabla = "Tabla_WR";
        document.getElementById("td_Objeto").innerHTML = Objeto;
        document.getElementById("td_Pared").innerHTML = Pared;
        document.getElementById("td_Largo").innerHTML = Largo;
        document.getElementById("td_Ancho").innerHTML = Ancho;
        document.getElementById("td_Superficie").innerHTML = Superficie;
        document.getElementById("td_Material").innerHTML = Material_WR.value;
        document.getElementById("add_objeto").disabled = false;
               
})

document.getElementById("add_objeto").disabled = true;

const TablaRT60 = document.getElementById('Tabla');
const GraficaRT60 = document.getElementById('Grafica');

document.getElementById('L2').style.display = "none";

TablaRT60.addEventListener('click',()=>{
    document.getElementById('L1').style.display = "block";
    document.getElementById('L2').style.display = "none";
})

GraficaRT60.addEventListener('click',()=>{
    document.getElementById('L2').style.display = "block";
    document.getElementById('L1').style.display = "none";
})

