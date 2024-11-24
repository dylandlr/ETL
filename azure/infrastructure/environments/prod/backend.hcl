terraform {
  backend "azurerm" {
    resource_group_name  = "terraform-state-rg"
    storage_account_name = "tfstate"
    container_name      = "state"
    key                 = "dev.terraform.tfstate"
  }
}